"""
reference_ranges.py  –  Built-in reference ranges derived from the
Blood Test Reference Ranges.pdf bundled in data/raw/.

These are used ONLY as a secondary fallback when the uploaded lab report
does NOT contain its own reference range column.
When a range IS printed on the report, that printed range always takes
precedence and this module is not invoked for that test.

⚠️  Important notices printed alongside every comparison:
    - Reference ranges vary by laboratory, method, age, and sex.
    - These values are sourced from the bundled PDF (general adult ranges).
    - Results must be interpreted by a qualified healthcare professional.
    - This tool cannot diagnose or rule out any condition.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Reference range table (adult, general population)
# Keys are normalised lowercase test names.
# Each entry: (low, high, unit, note)
# ---------------------------------------------------------------------------

# fmt: off
_RANGES: dict[str, tuple[float, float, str, str]] = {
    # Haematology
    "hemoglobin":                  (12.0,  17.5, "g/dL",      "Male 13.5–17.5 | Female 12.0–15.5"),
    "haemoglobin":                 (12.0,  17.5, "g/dL",      "Male 13.5–17.5 | Female 12.0–15.5"),
    "hgb":                         (12.0,  17.5, "g/dL",      "Male 13.5–17.5 | Female 12.0–15.5"),
    "rbc":                         (3.8,   5.8,  "10⁶/µL",   "Male 4.5–5.9 | Female 3.8–5.2"),
    "rbc count":                   (3.8,   5.8,  "10⁶/µL",   "Male 4.5–5.9 | Female 3.8–5.2"),
    "rbc_count":                   (3.8,   5.8,  "10⁶/µL",   "Male 4.5–5.9 | Female 3.8–5.2"),
    "wbc":                         (4.0,   11.0, "10³/µL",   ""),
    "wbc count":                   (4.0,   11.0, "10³/µL",   ""),
    "wbc_count":                   (4.0,   11.0, "10³/µL",   ""),
    "platelet count":              (150.0, 400.0,"10³/µL",   ""),
    "platelet_count":              (150.0, 400.0,"10³/µL",   ""),
    "platelets":                   (150.0, 400.0,"10³/µL",   ""),
    "hematocrit":                  (36.0,  52.0, "%",          "Male 41–53 | Female 36–46"),
    "haematocrit":                 (36.0,  52.0, "%",          "Male 41–53 | Female 36–46"),
    "mcv":                         (80.0,  100.0,"fL",         ""),
    "mch":                         (27.0,  33.0, "pg",         ""),
    "mchc":                        (32.0,  36.0, "g/dL",      ""),
    # Biochemistry
    "glucose":                     (70.0,  99.0, "mg/dL",     "Fasting"),
    "fasting blood sugar":         (70.0,  99.0, "mg/dL",     "Fasting"),
    "fasting_blood_sugar":         (70.0,  99.0, "mg/dL",     "Fasting"),
    "total cholesterol":           (0.0,   200.0,"mg/dL",     "Desirable < 200"),
    "total_cholesterol":           (0.0,   200.0,"mg/dL",     "Desirable < 200"),
    "cholesterol":                 (0.0,   200.0,"mg/dL",     "Desirable < 200"),
    "cholestrol":                  (0.0,   200.0,"mg/dL",     "Desirable < 200"),
    "ldl":                         (0.0,   100.0,"mg/dL",     "Optimal < 100 mg/dL (for low-risk adults)"),
    "hdl":                         (40.0,  9999.0,"mg/dL",   "Male ≥ 40 | Female ≥ 50; higher is better"),
    "triglycerides":               (0.0,   150.0,"mg/dL",     "Normal < 150"),
    "creatinine":                  (0.6,   1.2,  "mg/dL",     "Male 0.7–1.3 | Female 0.6–1.1"),
    "ast":                         (10.0,  40.0, "U/L",        ""),
    "ast (aspartate aminotransferase)": (10.0, 40.0, "U/L",   ""),
    "alt":                         (7.0,   56.0, "U/L",        ""),
    "alt (alanine aminotransferase)":   (7.0, 56.0, "U/L",    ""),
    "troponin":                    (0.0,   0.04, "ng/mL",     "Values ≥ 0.04 may indicate cardiac injury; confirm with clinical context"),
    "lipase":                      (0.0,   160.0,"U/L",        ""),
    "spirometry":                  (0.7,   1.2,  "FEV1/FVC",  "FEV1/FVC ratio; < 0.70 may suggest obstruction"),
}
# fmt: on

_DISCLAIMER = (
    "⚠️ Reference ranges vary by laboratory, patient sex, age, and testing method. "
    "These ranges are general adult values from the bundled reference PDF. "
    "Always use the range printed on your own laboratory report. "
    "Results must be interpreted by a qualified healthcare professional. "
    "This tool cannot diagnose or rule out any medical condition."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup(test_name: str) -> Optional[dict]:
    """
    Return the reference range entry for a test name, or None if not found.
    Returned dict: { low, high, unit, note, disclaimer }
    """
    key = _normalise(test_name)
    entry = _RANGES.get(key)
    if entry is None:
        return None
    low, high, unit, note = entry
    return {
        "low":        low,
        "high":       high,
        "unit":       unit,
        "note":       note,
        "disclaimer": _DISCLAIMER,
    }


def compare(test_name: str, value: float) -> dict:
    """
    Compare a numeric value against the built-in reference range.
    Always includes the disclaimer.

    Returns dict: { found, flag, low, high, unit, note, disclaimer }
    """
    ref = lookup(test_name)
    if ref is None:
        return {
            "found":       False,
            "flag":        "",
            "low":         None,
            "high":        None,
            "unit":        None,
            "note":        "",
            "disclaimer":  _DISCLAIMER,
        }

    flag = ""
    if value < ref["low"]:
        flag = "LOW"
    elif ref["high"] < 9000 and value > ref["high"]:
        flag = "HIGH"

    return {
        "found":      True,
        "flag":       flag,
        "low":        ref["low"],
        "high":       ref["high"],
        "unit":       ref["unit"],
        "note":       ref["note"],
        "disclaimer": _DISCLAIMER,
    }


def all_test_names() -> list[str]:
    """Return canonical list of test names the module knows about."""
    return sorted(set(_RANGES.keys()))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    import re
    return re.sub(r"\s+", " ", name.strip().lower())
