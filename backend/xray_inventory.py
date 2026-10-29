"""
xray_inventory.py  –  TB Chest Radiography Database inventory utility.

Supports the TB_Chest_Radiography_Database layout:
    data/raw/TB_Chest_Radiography_Database/
        Normal/        Normal-1.png … Normal-NNNN.png
        Tuberculosis/  Tuberculosis-1.png … Tuberculosis-NNN.png
        Normal.metadata.xlsx
        Tuberculosis.metadata.xlsx
        README.md.txt

Label is parsed from the directory name, NOT a filename suffix,
because this dataset uses directory-based labelling.

Returns a rich inventory dict; raises InventoryError on fatal problems.
"""

from __future__ import annotations

import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# Expected database layout (relative to project root)
# ---------------------------------------------------------------------------
DB_RELPATH   = os.path.join("data", "raw", "TB_Chest_Radiography_Database")
NORMAL_SUBDIR = "Normal"
TB_SUBDIR     = "Tuberculosis"
IMAGE_EXTS    = {".png", ".jpg", ".jpeg"}

# Shenzhen-style suffix pattern (_0 = normal, _1 = TB) — supported as
# an alternative if the folder ever changes to that naming scheme.
_SUFFIX_RE = re.compile(r"_([01])$", re.IGNORECASE)


class InventoryError(Exception):
    """Raised when the dataset is missing or has a fatal layout problem."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inventory(project_root: str) -> dict[str, Any]:
    """
    Scan the TB_Chest_Radiography_Database and return an inventory dict.

    Dict keys:
        db_path          – absolute path inspected
        db_exists        – bool
        error            – str | None  (fatal error message)
        warnings         – list[str]
        normal_count     – int
        tb_count         – int
        total_count      – int
        unreadable       – list[str]  (filenames that caused errors)
        label_source     – 'directory' | 'suffix' | 'unknown'
        summary          – human-readable one-liner
    """
    result: dict[str, Any] = {
        "db_path":       "",
        "db_exists":     False,
        "error":         None,
        "warnings":      [],
        "normal_count":  0,
        "tb_count":      0,
        "total_count":   0,
        "unreadable":    [],
        "label_source":  "unknown",
        "summary":       "",
    }

    db_path = os.path.normpath(os.path.join(project_root, DB_RELPATH))
    result["db_path"] = db_path

    if not os.path.isdir(db_path):
        result["error"] = (
            f"TB Chest Radiography Database NOT FOUND at expected path:\n"
            f"  {db_path}\n\n"
            f"Expected directory structure:\n"
            f"  {db_path}/\n"
            f"    {NORMAL_SUBDIR}/      ← Normal chest X-rays\n"
            f"    {TB_SUBDIR}/  ← TB chest X-rays\n"
            f"    Normal.metadata.xlsx\n"
            f"    Tuberculosis.metadata.xlsx\n\n"
            f"X-ray inventory cannot be built. Please add the dataset before "
            f"re-running the inventory."
        )
        return result

    result["db_exists"] = True

    normal_dir = os.path.join(db_path, NORMAL_SUBDIR)
    tb_dir     = os.path.join(db_path, TB_SUBDIR)

    # Both subdirectories must exist
    missing = []
    if not os.path.isdir(normal_dir):
        missing.append(f"  {normal_dir}  (Normal images)")
    if not os.path.isdir(tb_dir):
        missing.append(f"  {tb_dir}  (TB images)")
    if missing:
        result["error"] = (
            "Required sub-directories are missing:\n"
            + "\n".join(missing)
            + "\nX-ray inventory aborted."
        )
        return result

    # Scan each directory
    normal_count, normal_unreadable = _scan_dir(normal_dir)
    tb_count,     tb_unreadable     = _scan_dir(tb_dir)

    result["normal_count"] = normal_count
    result["tb_count"]     = tb_count
    result["total_count"]  = normal_count + tb_count
    result["unreadable"]   = normal_unreadable + tb_unreadable
    result["label_source"] = "directory"

    if normal_unreadable:
        result["warnings"].append(
            f"{len(normal_unreadable)} file(s) in {NORMAL_SUBDIR}/ "
            f"could not be read: {normal_unreadable[:5]}"
        )
    if tb_unreadable:
        result["warnings"].append(
            f"{len(tb_unreadable)} file(s) in {TB_SUBDIR}/ "
            f"could not be read: {tb_unreadable[:5]}"
        )

    if normal_count == 0 and tb_count == 0:
        result["error"] = (
            "Both Normal/ and Tuberculosis/ directories are empty (no image files). "
            "Place PNG/JPG images in each subdirectory before running inventory."
        )
        return result

    if normal_count == 0:
        result["warnings"].append(
            f"Normal/ directory is empty (0 readable images). "
            f"Only TB images were counted."
        )
    if tb_count == 0:
        result["warnings"].append(
            f"Tuberculosis/ directory is empty (0 readable images). "
            f"Only Normal images were counted."
        )

    result["summary"] = (
        f"TB_Chest_Radiography_Database — {result['total_count']} images total: "
        f"{normal_count} Normal  |  {tb_count} Tuberculosis"
    )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scan_dir(directory: str) -> tuple[int, list[str]]:
    """
    Count readable image files in *directory*.
    Returns (count, list_of_unreadable_filenames).
    """
    count = 0
    unreadable: list[str] = []

    try:
        entries = os.listdir(directory)
    except OSError as e:
        return 0, [f"<cannot list directory: {e}>"]

    for fname in entries:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTS:
            continue
        fpath = os.path.join(directory, fname)
        try:
            # Check file is readable and non-empty
            size = os.path.getsize(fpath)
            if size == 0:
                unreadable.append(fname)
            else:
                count += 1
        except OSError:
            unreadable.append(fname)

    return count, unreadable
