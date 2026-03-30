"""ToolRef RAG evaluation runner.

Usage (from repo root):
    # IR metrics only (no LLM judge, fast)
    python -m eval.run --mode ir

    # Full evaluation with RAGAS (requires LLM API)
    python -m eval.run --mode full

    # Specific test cases
    python -m eval.run --cases tc-001 tc-002 tc-003

Environment:
    TOOLREF_API_URL: ToolRef API base URL (default: http://localhost:8000)
    EVAL_LLM_MODEL: LLM model for RAGAS judge (default: uses app's configured LLM)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from eval.metrics import IRMetrics, aggregate_ir_metrics, compute_ir_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("eval")

EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "dataset.json"
RESULTS_DIR = EVAL_DIR / "results"

API_URL = os.environ.get("TOOLREF_API_URL", "http://localhost:8000")
QUERY_ENDPOINT = f"{API_URL}/api/v1/query"
QUERY_TIMEOUT = 300  # seconds — CRAG loops can be slow on CPU


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    """A single evaluation test case."""

    id: str
    query: str
    expected_doc_titles: list[str]
    ground_truth: str
    category: str = "factual"
    difficulty: str = "easy"
    note: str = ""


@dataclass
class EvalResult:
    """Result for a single test case."""

    test_case_id: str
    query: str
    category: str
    difficulty: str

    # RAG response
    answer: str = ""
    retrieved_doc_titles: list[str] = field(default_factory=list)
    rewrite_count: int = 0
    latency_ms: int = 0
    cached: bool = False

    # IR metrics
    ir_metrics: dict[str, float] = field(default_factory=dict)

    # RAGAS metrics (None if not computed)
    ragas_metrics: dict[str, float] | None = None

    # Error
    error: str | None = None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(
    path: Path = DATASET_PATH,
    case_ids: list[str] | None = None,
) -> list[TestCase]:
    """Load evaluation dataset, optionally filtering by case IDs."""
    with open(path) as f:
        data = json.load(f)

    cases = []
    for tc in data["test_cases"]:
        if case_ids and tc["id"] not in case_ids:
            continue
        cases.append(TestCase(
            id=tc["id"],
            query=tc["query"],
            expected_doc_titles=tc["expected_doc_titles"],
            ground_truth=tc.get("ground_truth", ""),
            category=tc.get("category", "factual"),
            difficulty=tc.get("difficulty", "easy"),
            note=tc.get("note", ""),
        ))

    logger.info("Loaded %d test cases from %s", len(cases), path)
    return cases


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------
def run_query(query: str, namespace: str = "default") -> dict[str, Any]:
    """Execute a single query against ToolRef API (non-streaming)."""
    with httpx.Client(timeout=QUERY_TIMEOUT) as client:
        resp = client.post(
            QUERY_ENDPOINT,
            json={"query": query, "namespace": namespace},
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_case(tc: TestCase, namespace: str = "default") -> EvalResult:
    """Run a single test case through the RAG pipeline and compute metrics."""
    result = EvalResult(
        test_case_id=tc.id,
        query=tc.query,
        category=tc.category,
        difficulty=tc.difficulty,
    )

    try:
        t0 = time.monotonic()
        response = run_query(tc.query, namespace)
        elapsed = int((time.monotonic() - t0) * 1000)

        result.answer = response.get("answer", "")
        result.latency_ms = response.get("latency_ms", elapsed)
        result.rewrite_count = response.get("rewrite_count", 0)
        result.cached = response.get("cached", False)

        # Extract retrieved doc titles
        sources = response.get("sources", [])
        # Deduplicate while preserving order
        seen: set[str] = set()
        for s in sources:
            title = s.get("doc_title", "")
            if title and title not in seen:
                result.retrieved_doc_titles.append(title)
                seen.add(title)

        # IR metrics
        ir = compute_ir_metrics(
            retrieved_doc_titles=result.retrieved_doc_titles,
            expected_doc_titles=tc.expected_doc_titles,
            k=5,
        )
        result.ir_metrics = asdict(ir)

    except Exception as e:
        result.error = str(e)
        logger.exception("Failed to evaluate %s", tc.id)

    return result


def run_ragas_evaluation(
    results: list[EvalResult],
    test_cases: list[TestCase],
) -> list[EvalResult]:
    """Run RAGAS metrics on evaluation results.

    Requires: pip install ragas
    Uses the configured LLM for judge evaluation.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
        )
    except ImportError:
        logger.warning(
            "RAGAS not installed — skipping RAGAS metrics. "
            "Install with: pip install ragas"
        )
        return results

    # Build RAGAS dataset
    samples = []
    valid_indices = []

    for i, (result, tc) in enumerate(zip(results, test_cases)):
        if result.error:
            continue
        if tc.category == "out_of_scope":
            continue

        # RAGAS expects: user_input, response, retrieved_contexts, reference
        response_data = result.answer or ""
        retrieved_contexts = [
            f"[{title}]" for title in result.retrieved_doc_titles
        ]
        # For reference-free metrics, ground_truth is optional but helps recall
        reference = tc.ground_truth or ""

        sample = SingleTurnSample(
            user_input=tc.query,
            response=response_data,
            retrieved_contexts=retrieved_contexts if retrieved_contexts else ["(no context retrieved)"],
            reference=reference if reference else None,
        )
        samples.append(sample)
        valid_indices.append(i)

    if not samples:
        logger.warning("No valid samples for RAGAS evaluation")
        return results

    logger.info("Running RAGAS evaluation on %d samples...", len(samples))

    try:
        dataset = EvaluationDataset(samples=samples)

        # Select metrics based on whether we have ground truth
        metrics = [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
        ]

        # Only add recall if we have ground truth references
        has_references = any(s.reference for s in samples)
        if has_references:
            metrics.append(LLMContextRecall())

        ragas_result = ragas_evaluate(dataset=dataset, metrics=metrics)

        # Map scores back to results
        scores_df = ragas_result.to_pandas()
        for j, idx in enumerate(valid_indices):
            if j < len(scores_df):
                row = scores_df.iloc[j]
                results[idx].ragas_metrics = {
                    col: float(row[col])
                    for col in scores_df.columns
                    if col not in ("user_input", "response", "retrieved_contexts", "reference")
                    and not str(row[col]) == "nan"
                }

        logger.info("RAGAS evaluation complete")

    except Exception:
        logger.exception("RAGAS evaluation failed")

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(results: list[EvalResult]) -> None:
    """Print evaluation report to terminal."""
    print("\n" + "=" * 70)
    print("  ToolRef RAG Evaluation Report")
    print("=" * 70)

    # Per-case results
    for r in results:
        status = "✅" if r.error is None else "❌"
        hit = r.ir_metrics.get("hit", False)
        hit_icon = "🎯" if hit else "⭕"
        rr = r.ir_metrics.get("reciprocal_rank", 0)

        print(f"\n{status} {r.test_case_id} [{r.category}/{r.difficulty}]")
        print(f"   Query: {r.query[:70]}...")
        print(f"   {hit_icon} Hit={hit}  RR={rr:.2f}  Rewrites={r.rewrite_count}  Latency={r.latency_ms}ms")
        print(f"   Retrieved: {r.retrieved_doc_titles or '(none)'}")

        if r.ragas_metrics:
            metrics_str = "  ".join(f"{k}={v:.3f}" for k, v in r.ragas_metrics.items())
            print(f"   RAGAS: {metrics_str}")

        if r.error:
            print(f"   ❌ Error: {r.error}")

    # Aggregate IR metrics
    ir_results = [
        IRMetrics(**r.ir_metrics) for r in results
        if r.ir_metrics and r.error is None
    ]
    if ir_results:
        agg = aggregate_ir_metrics(ir_results)
        print("\n" + "-" * 70)
        print("  Aggregate IR Metrics")
        print("-" * 70)
        print(f"  Hit Rate@5:    {agg['hit_rate']:.1%}")
        print(f"  MRR:           {agg['mrr']:.3f}")
        print(f"  Precision@5:   {agg['precision_at_k']:.3f}")
        print(f"  Recall@5:      {agg['recall_at_k']:.3f}")

    # Aggregate RAGAS metrics
    ragas_results = [r.ragas_metrics for r in results if r.ragas_metrics]
    if ragas_results:
        print("\n" + "-" * 70)
        print("  Aggregate RAGAS Metrics")
        print("-" * 70)
        all_keys = set()
        for rm in ragas_results:
            all_keys.update(rm.keys())
        for key in sorted(all_keys):
            values = [rm[key] for rm in ragas_results if key in rm]
            if values:
                avg = sum(values) / len(values)
                print(f"  {key}: {avg:.3f} (n={len(values)})")

    print("\n" + "=" * 70)


def save_report(results: list[EvalResult], mode: str) -> Path:
    """Save evaluation results as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"eval_{mode}_{timestamp}.json"

    # Compute aggregates
    ir_results = [
        IRMetrics(**r.ir_metrics) for r in results
        if r.ir_metrics and r.error is None
    ]
    agg_ir = aggregate_ir_metrics(ir_results) if ir_results else {}

    ragas_results = [r.ragas_metrics for r in results if r.ragas_metrics]
    agg_ragas: dict[str, float] = {}
    if ragas_results:
        all_keys: set[str] = set()
        for rm in ragas_results:
            all_keys.update(rm.keys())
        for key in sorted(all_keys):
            values = [rm[key] for rm in ragas_results if key in rm]
            if values:
                agg_ragas[key] = sum(values) / len(values)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "total_cases": len(results),
        "errors": sum(1 for r in results if r.error),
        "aggregate_ir": agg_ir,
        "aggregate_ragas": agg_ragas,
        "results": [
            {
                "test_case_id": r.test_case_id,
                "query": r.query,
                "category": r.category,
                "difficulty": r.difficulty,
                "answer_preview": r.answer[:200] if r.answer else "",
                "retrieved_doc_titles": r.retrieved_doc_titles,
                "rewrite_count": r.rewrite_count,
                "latency_ms": r.latency_ms,
                "cached": r.cached,
                "ir_metrics": r.ir_metrics,
                "ragas_metrics": r.ragas_metrics,
                "error": r.error,
            }
            for r in results
        ],
    }

    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Report saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="ToolRef RAG Evaluation")
    parser.add_argument(
        "--mode",
        choices=["ir", "full"],
        default="ir",
        help="'ir' = IR metrics only (fast), 'full' = IR + RAGAS (requires LLM)",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        help="Specific test case IDs to run (e.g., tc-001 tc-002)",
    )
    parser.add_argument(
        "--namespace",
        default="default",
        help="Namespace to query (default: default)",
    )
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("  ToolRef Evaluation — mode=%s", args.mode)
    logger.info("  API: %s", API_URL)
    logger.info("  Namespace: %s", args.namespace)
    logger.info("=" * 50)

    # Load dataset
    test_cases = load_dataset(case_ids=args.cases)
    if not test_cases:
        logger.error("No test cases to evaluate")
        sys.exit(1)

    # Run evaluations
    results: list[EvalResult] = []
    for i, tc in enumerate(test_cases, 1):
        logger.info("[%d/%d] Evaluating %s: %s", i, len(test_cases), tc.id, tc.query[:60])
        result = evaluate_case(tc, namespace=args.namespace)
        results.append(result)
        if result.error:
            logger.error("  ❌ %s: %s", tc.id, result.error)
        else:
            hit = result.ir_metrics.get("hit", False)
            logger.info(
                "  → hit=%s rr=%.2f rewrites=%d latency=%dms",
                hit, result.ir_metrics.get("reciprocal_rank", 0),
                result.rewrite_count, result.latency_ms,
            )

    # RAGAS evaluation (if full mode)
    if args.mode == "full":
        results = run_ragas_evaluation(results, test_cases)

    # Report
    print_report(results)
    report_path = save_report(results, args.mode)
    print(f"\n📄 Report saved: {report_path}")


if __name__ == "__main__":
    main()
