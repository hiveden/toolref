"""Document parsing module using Unstructured.io.

Converts raw file bytes into a list of :class:`DocumentElement` objects,
preserving structural metadata such as page numbers and heading levels.
"""

import logging
import tempfile
from dataclasses import dataclass, field
from typing import Any

from app.db.models import DocType

logger = logging.getLogger(__name__)


@dataclass
class DocumentElement:
    """A structural element extracted from a parsed document.

    Attributes:
        element_type: Semantic type — e.g. ``"paragraph"``, ``"heading"``,
            ``"table"``, ``"code_block"``, ``"list_item"``.
        text: Plain-text content of the element.
        metadata: Extra info such as ``page_number``, ``heading_level``, etc.
    """

    element_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _map_element_type(element: Any) -> str:
    """Map an Unstructured element class name to a simplified type string."""
    class_name = type(element).__name__
    mapping: dict[str, str] = {
        "Title": "heading",
        "Header": "heading",
        "NarrativeText": "paragraph",
        "ListItem": "list_item",
        "Table": "table",
        "FigureCaption": "figure_caption",
        "Image": "image",
        "Address": "address",
        "EmailAddress": "email",
        "Formula": "formula",
        "CodeSnippet": "code_block",
    }
    return mapping.get(class_name, "paragraph")


def _extract_metadata(element: Any) -> dict[str, Any]:
    """Pull useful metadata from an Unstructured element."""
    meta: dict[str, Any] = {}
    el_meta = getattr(element, "metadata", None)
    if el_meta is None:
        return meta

    if hasattr(el_meta, "page_number") and el_meta.page_number is not None:
        meta["page_number"] = el_meta.page_number
    if hasattr(el_meta, "category_depth") and el_meta.category_depth is not None:
        meta["heading_level"] = el_meta.category_depth
    if hasattr(el_meta, "filename") and el_meta.filename is not None:
        meta["filename"] = el_meta.filename

    return meta


def parse_document(
    file_bytes: bytes,
    doc_type: DocType,
    metadata: dict[str, Any] | None = None,
) -> list[DocumentElement]:
    """Parse raw file bytes into a list of :class:`DocumentElement`.

    Uses Unstructured.io partitioners for PDF, Markdown, and HTML.
    Plain-text files are handled directly without Unstructured.

    Args:
        file_bytes: Raw bytes of the document.
        doc_type: The type/format of the document.
        metadata: Optional extra metadata to attach to every element.

    Returns:
        A list of ``DocumentElement`` instances.  Returns an empty list
        when parsing fails (error is logged).
    """
    extra_meta = metadata or {}

    try:
        if doc_type == DocType.TXT:
            return _parse_txt(file_bytes, extra_meta)
        return _parse_with_unstructured(file_bytes, doc_type, extra_meta)
    except Exception:
        logger.exception("Failed to parse document (type=%s)", doc_type.value)
        return []


def _parse_txt(file_bytes: bytes, extra_meta: dict[str, Any]) -> list[DocumentElement]:
    """Handle plain-text files by splitting on double newlines."""
    text = file_bytes.decode("utf-8", errors="replace")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    elements: list[DocumentElement] = []
    for idx, para in enumerate(paragraphs):
        elements.append(
            DocumentElement(
                element_type="paragraph",
                text=para,
                metadata={**extra_meta, "paragraph_index": idx},
            )
        )
    logger.info("Parsed TXT document: %d elements", len(elements))
    return elements


def _parse_with_unstructured(
    file_bytes: bytes,
    doc_type: DocType,
    extra_meta: dict[str, Any],
) -> list[DocumentElement]:
    """Parse a document using the appropriate Unstructured partitioner."""

    # Write bytes to a temporary file — Unstructured needs a file path.
    suffix_map: dict[DocType, str] = {
        DocType.PDF: ".pdf",
        DocType.MARKDOWN: ".md",
        DocType.HTML: ".html",
    }
    suffix = suffix_map.get(doc_type, ".bin")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name

        if doc_type == DocType.PDF:
            from unstructured.partition.pdf import partition_pdf

            raw_elements = partition_pdf(filename=tmp_path, strategy="hi_res")
        elif doc_type == DocType.MARKDOWN:
            from unstructured.partition.md import partition_md

            raw_elements = partition_md(filename=tmp_path)
        elif doc_type == DocType.HTML:
            from unstructured.partition.html import partition_html

            raw_elements = partition_html(filename=tmp_path)
        else:
            logger.warning("Unsupported doc_type '%s', falling back to TXT", doc_type.value)
            return _parse_txt(file_bytes, extra_meta)

    elements: list[DocumentElement] = []
    for el in raw_elements:
        text = str(el).strip()
        if not text:
            continue
        elements.append(
            DocumentElement(
                element_type=_map_element_type(el),
                text=text,
                metadata={**extra_meta, **_extract_metadata(el)},
            )
        )

    logger.info("Parsed %s document: %d elements", doc_type.value, len(elements))
    return elements
