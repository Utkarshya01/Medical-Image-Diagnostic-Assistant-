"""
lab_parser.py  –  Local PDF lab-report extraction module.

Rules enforced:
- Never guess a missing value / unit / range.
- Mark uncertain fields "Please check the original report."
- Never infer diseases or recommend medication.
- No data persisted to disk; everything stays in RAM.
- If the PDF is image-only (scanned), return an OCR-required status
  unless pytesseract + Tesseract binary are available locally.
"""

from __future__ import annotations

import os
import io
import re
from typing import Any

# ---------------------------------------------------------------------------
# Optional imports — handle gracefully if packages are not installed yet.
# ---------------------------------------------------------------------------

try:
    import pdfplumber
    _PDFPLUMBER_OK = True
except ImportError:
    _PDFPLUMBER_OK = False

try:
    import pytesseract
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    _PYTESSERACT_AVAILABLE = False

def check_tesseract() -> bool:
    """Check if Tesseract binary is accessible, setting standard Windows path if found."""
    global pytesseract
    try:
        import shutil
        if pytesseract is None:
            import pytesseract as _pt
            pytesseract = _pt
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
        ]
        cmd = shutil.which("tesseract")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
            tessdata = os.path.join(os.path.dirname(cmd), "tessdata")
            if os.path.isdir(tessdata) and "TESSDATA_PREFIX" not in os.environ:
                os.environ["TESSDATA_PREFIX"] = tessdata
        else:
            for cand in candidates:
                if os.path.isfile(cand):
                    pytesseract.pytesseract.tesseract_cmd = cand
                    tessdata = os.path.join(os.path.dirname(cand), "tessdata")
                    if os.path.isdir(tessdata) and "TESSDATA_PREFIX" not in os.environ:
                        os.environ["TESSDATA_PREFIX"] = tessdata
                    break
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

_TESSERACT_OK = check_tesseract()



# ---------------------------------------------------------------------------
# Constants – patterns for common lab result lines
# ---------------------------------------------------------------------------

# Matches:  "Hemoglobin   13.5   g/dL   11.5–18.0   H"
# Groups:    test_name  |  value  | unit |  range  | flag?
_ROW_RE = re.compile(
    r"([A-Za-z][\w\s/(),\-\.]{1,60?}?)"   # test name (non-greedy)
    r"\s{2,}"                               # two+ spaces (column separator)
    r"([0-9]+(?:\.[0-9]+)?)"               # numeric value
    r"\s*"
    r"([A-Za-z/%µ][A-Za-z0-9/%µ·\-]*)"    # unit
    r"\s{2,}"
    r"([0-9]+(?:\.[0-9]+)?[\s\-–—to]+[0-9]+(?:\.[0-9]+)?)" # range
    r"(?:\s*(H|L|High|Low|Abnormal|Above|Below|Critical))?"  # optional flag
    ,
    re.IGNORECASE,
)

# Simpler line pattern when column alignment is tighter
_LINE_RE = re.compile(
    r"([A-Za-z][\w\s(),\-\.]{2,50}?)"
    r"[\s:]+([0-9]+(?:[.,][0-9]+)?)"
    r"\s*([A-Za-z/%µ][A-Za-z0-9/%µ·\-]*)"
    r"(?:\s+(?:Ref(?:erence)?\.?\s*(?:Range)?|Normal)?\.?:?\s*"
    r"([0-9]+(?:\.[0-9]+)?[\s\-–—to]+[0-9]+(?:\.[0-9]+)?))?"
    r"(?:\s*(H|L|High|Low|\*))?"
    , re.IGNORECASE,
)

_UNCERTAIN = "Please check the original report."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(file_bytes: bytes, filename: str = "report.pdf") -> dict[str, Any]:
    """
    Parse a lab-report PDF from raw bytes.

    Returns a dict with:
        raw_text   – full extracted text (str | None)
        rows       – list of structured row dicts
        ocr_used   – bool
        ocr_available – bool (Tesseract found)
        warnings   – list of warning strings
        error      – str | None
    """
    tess_ok = check_tesseract()
    result: dict[str, Any] = {
        "raw_text": None,
        "rows": [],
        "ocr_used": False,
        "ocr_available": tess_ok,
        "warnings": [],
        "error": None,
    }

    if not _PDFPLUMBER_OK:
        result["error"] = (
            "pdfplumber is not installed. "
            "Run: pip install pdfplumber"
        )
        return result

    try:
        pdf_file = io.BytesIO(file_bytes)
        raw_pages: list[str] = []
        image_pages: list[int] = []

        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    raw_pages.append(text)
                else:
                    image_pages.append(page_num)
                    # Try pdfplumber table extraction as fallback
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            for row in table:
                                if row:
                                    line = "  ".join(
                                        (cell or "").strip() for cell in row
                                    )
                                    raw_pages.append(line)

        full_text = "\n".join(raw_pages).strip()

        # If NO text at all — it's a scanned PDF
        if not full_text:
            if image_pages:
                if tess_ok:
                    result["warnings"].append(
                        f"Page(s) {image_pages} appear to be scanned images. "
                        "Attempting OCR with local Tesseract."
                    )
                    ocr_text = _run_ocr_on_pdf(file_bytes)
                    if ocr_text:
                        full_text = ocr_text
                        result["ocr_used"] = True
                    else:
                        result["error"] = (
                            "This scanned PDF needs OCR support. "
                            "Tesseract was found but produced no text."
                        )
                        return result
                else:
                    result["error"] = (
                        "This scanned PDF needs OCR support. "
                        "Local Tesseract is NOT installed or not on PATH. "
                        "Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki "
                        "then run: pip install pytesseract"
                    )
                    return result
        elif image_pages:
            result["warnings"].append(
                f"Page(s) {image_pages} appear to be image-only. "
                f"Text was extracted from other pages; image pages were skipped."
            )

        result["raw_text"] = full_text
        result["rows"] = _extract_rows(full_text)

        if not result["rows"]:
            result["warnings"].append(
                "No structured lab rows were automatically detected. "
                "Please review the extracted text above for your values."
            )

    except Exception as exc:  # noqa: BLE001
        result["error"] = f"PDF parsing error: {exc}"

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_ocr_on_pdf(file_bytes: bytes) -> str:
    """Run Tesseract OCR on every page rendered as a PIL image."""
    if not check_tesseract():
        return ""
    try:
        import pdfplumber  # already checked but be safe
        pages_text: list[str] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                # render to PIL image via pdfplumber's to_image
                img_obj = page.to_image(resolution=200)
                pil_img = img_obj.original
                ocr_text = pytesseract.image_to_string(pil_img, lang="eng")
                if ocr_text.strip():
                    pages_text.append(ocr_text)
        return "\n".join(pages_text)
    except Exception:  # noqa: BLE001
        return ""


def _extract_rows(text: str) -> list[dict[str, str]]:
    """
    Heuristically extract structured rows from lab report text.
    Only returns rows with at least test_name and value.
    Uncertain fields are marked explicitly.
    """
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for line in text.splitlines():
        line = line.strip()
        if len(line) < 5:
            continue

        row = _parse_line(line)
        if row is None:
            continue

        # Deduplicate
        key = (row["test_name"].lower(), row["value"])
        if key in seen:
            continue
        seen.add(key)

        rows.append(row)

    return rows


def _parse_line(line: str) -> dict[str, str] | None:
    """Try to parse a single line into a structured row."""
    # Attempt strict multi-column regex first
    m = _ROW_RE.search(line)
    if m:
        name, value, unit, ref_range, flag = m.groups()
        return {
            "test_name":   _clean(name),
            "value":       value.strip(),
            "unit":        _clean(unit) if unit else _UNCERTAIN,
            "ref_range":   _clean(ref_range) if ref_range else _UNCERTAIN,
            "flag":        _normalise_flag(flag),
            "source_line": line,
        }

    # Looser single-line regex
    m2 = _LINE_RE.search(line)
    if m2:
        name, value, unit, ref_range, flag = m2.groups()
        # Filter noise: skip very short "names" that are likely fragments
        if len((name or "").strip()) < 3:
            return None
        return {
            "test_name":   _clean(name),
            "value":       value.strip().replace(",", "."),
            "unit":        _clean(unit) if unit else _UNCERTAIN,
            "ref_range":   _clean(ref_range) if ref_range else _UNCERTAIN,
            "flag":        _normalise_flag(flag),
            "source_line": line,
        }

    return None


def _clean(s: str | None) -> str:
    if not s:
        return _UNCERTAIN
    return re.sub(r"\s{2,}", " ", s).strip(" :\t,")


def _normalise_flag(flag: str | None) -> str:
    if not flag:
        return ""
    f = flag.strip().upper()
    if f in {"H", "HIGH", "ABOVE"}:
        return "HIGH"
    if f in {"L", "LOW", "BELOW"}:
        return "LOW"
    if f in {"*", "ABNORMAL", "CRITICAL"}:
        return "ABNORMAL"
    return flag.strip()


# ---------------------------------------------------------------------------
# Image (not PDF) parsing — accept PNG/JPG lab reports via Tesseract
# ---------------------------------------------------------------------------

def parse_image(file_bytes: bytes, filename: str = "report.png") -> dict[str, Any]:
    """
    Parse a scanned lab report from an image file using Tesseract OCR.
    """
    tess_ok = check_tesseract()
    result: dict[str, Any] = {
        "raw_text": None,
        "rows": [],
        "ocr_used": False,
        "ocr_available": tess_ok,
        "warnings": [],
        "error": None,
    }

    if not tess_ok:
        result["error"] = (
            "This scanned PDF needs OCR support. "
            "Local Tesseract is NOT installed or not on PATH. "
            "Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki "
            "then run: pip install pytesseract"
        )
        return result

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img, lang="eng")
        if not text.strip():
            result["error"] = "OCR produced no text from this image."
            return result
        result["raw_text"] = text
        result["ocr_used"] = True
        result["rows"] = _extract_rows(text)
        result["warnings"].append(
            "OCR was used to extract text from this image. "
            "Please verify all values against the original document."
        )
        if not result["rows"]:
            result["warnings"].append(
                "No structured lab rows were automatically detected. "
                "Review the extracted text for your values."
            )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"Image OCR error: {exc}"

    return result
