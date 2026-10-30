"""
routes.py – Flask Blueprint for Healthcare Diagnostic Assistant
Stage 2: Lab-report parsing, X-ray inventory, dataset comparison.

⚠️  DISCLAIMER
This application is a research/demo screening aid only.
It is NOT a medical device and must NOT be used for diagnosis,
treatment, or ruling out any condition.
Always consult a qualified healthcare professional.
"""

from __future__ import annotations

import os
import csv
import json
import io
import re
from flask import Blueprint, jsonify, request, current_app, send_file

from backend.lab_parser      import parse_pdf, parse_image
from backend.xray_inventory  import run_inventory
from backend.reference_ranges import compare as ref_compare, _DISCLAIMER as REF_DISCLAIMER
from backend.xray_model      import (
    predict_chest_xray,
    validate_chest_xray,
    CLINICAL_DISCLAIMER,
    PATIENT_GROUPING_LIMITATION,
    MODEL_LIMITATIONS,
    CLASS_MAPPING,
)
from backend.chatbot         import get_chatbot_response

api = Blueprint("api", __name__, url_prefix="/api")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_IMG_EXT  = {"png", "jpg", "jpeg"}
ALLOWED_PDF_EXT  = {"pdf"}
MAX_PDF_BYTES    = 20 * 1024 * 1024   # 20 MB
MAX_IMG_BYTES    = 10 * 1024 * 1024   # 10 MB

_DISCLAIMER = (
    "Research/demo screening aid only. "
    "Not a medical device. "
    "Not for diagnosis, treatment, or ruling out any condition. "
    "Consult a qualified healthcare professional."
)


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


# ---------------------------------------------------------------------------
# 1. Health & Model Availability Status
# ---------------------------------------------------------------------------

@api.route("/health", methods=["GET"])
@api.route("/model/status", methods=["GET"])
@api.route("/xray/status", methods=["GET"])
def health():
    """
    Health check and model availability status.
    Verifies that the server is alive and reports whether the transfer-learning
    baseline model weights exist under artifacts/.
    """
    project_root = current_app.root_path
    weights_path = os.path.join(project_root, "artifacts", "tb_xray_baseline.pt")
    model_available = os.path.isfile(weights_path)

    return jsonify({
        "status":             "ok",
        "stage":              "Stage 2 – Local lab parsing + X-ray baseline active",
        "model_available":    model_available,
        "model_architecture": "resnet18",
        "threshold":          0.50,
        "classes":            ["Normal", "TB-consistent findings"],
        "disclaimer":         _DISCLAIMER,
    }), 200


# ---------------------------------------------------------------------------
# 2. Lab-report PDF/Image upload & in-memory parse
# ---------------------------------------------------------------------------

@api.route("/lab/parse", methods=["POST"])
@api.route("/lab/upload", methods=["POST"])
def lab_parse():
    """
    Accept a single PDF or image (PNG/JPG/JPEG) lab report.
    Validates file format, enforces size limits, and checks file content validity.
    Processed strictly in memory (RAM) and NEVER saved to disk.
    Does NOT log filenames, report text, or contents.

    Returns: extracted structured rows, report-printed reference ranges,
    flags, OCR status, and cautious plain-language explanations based ONLY
    on the report's own printed ranges.
    """
    f = request.files.get("file") or request.files.get("report")
    if not f or f.filename == "":
        return jsonify({"error": "No file selected or 'file' field missing in request."}), 400

    ext = _ext(f.filename)

    if ext not in ALLOWED_PDF_EXT | ALLOWED_IMG_EXT:
        return jsonify({
            "error": (
                f"Unsupported file type '{ext}'. "
                "Accepted: PDF, PNG, JPG, JPEG."
            )
        }), 415

    file_bytes = f.read()

    # Empty file validation
    if not file_bytes or len(file_bytes) == 0:
        return jsonify({"error": "Uploaded file is empty (0 bytes)."}), 400

    # Size limit validation
    if ext == "pdf" and len(file_bytes) > MAX_PDF_BYTES:
        return jsonify({"error": "PDF too large (max 20 MB)."}), 413

    if ext in ALLOWED_IMG_EXT and len(file_bytes) > MAX_IMG_BYTES:
        return jsonify({"error": "Image too large (max 10 MB)."}), 413

    # Content structure validation
    if ext == "pdf":
        if not file_bytes.startswith(b"%PDF-"):
            return jsonify({
                "error": "Invalid PDF content: file header is missing '%PDF-' signature or is corrupted."
            }), 400
        # In-memory parsing
        result = parse_pdf(file_bytes, filename="report.pdf")
    else:
        try:
            from PIL import Image
            img_check = Image.open(io.BytesIO(file_bytes))
            img_check.verify()
        except Exception as exc:
            return jsonify({"error": f"Invalid or unreadable image file: {exc}"}), 400
        # In-memory parsing
        result = parse_image(file_bytes, filename="report.png")

    if result.get("error") and not result.get("raw_text"):
        return jsonify({
            "error": result["error"],
            "warnings": result.get("warnings", []),
            "disclaimer": _DISCLAIMER,
        }), 400

    # Enrich rows with plain-language explanations based ONLY on the report's own ranges
    enriched_rows = []
    for row in result.get("rows", []):
        enriched = dict(row)
        # Cautious plain-language explanation based strictly on report's own ranges
        enriched["plain_explanation"] = _plain_explanation(enriched)
        enriched_rows.append(enriched)

    return jsonify({
        "size_bytes":     len(file_bytes),
        "ocr_used":       result.get("ocr_used", False),
        "ocr_available":  result.get("ocr_available", False),
        "raw_text":       result.get("raw_text"),
        "rows":           enriched_rows,
        "warnings":       result.get("warnings", []),
        "error":          result.get("error"),
        "disclaimer":     _DISCLAIMER,
    }), 200


def _plain_explanation(row: dict) -> str:
    """
    Generate a concise, very short plain-language explanation.
    Never infers diseases, never recommends medication, never calculates risk.
    """
    ref   = row.get("ref_range", "")
    flag  = (row.get("flag") or "").upper()

    is_ref_uncertain = not ref or ref == "Please check the original report." or ref == "—" or ref == "Not specified"

    if is_ref_uncertain:
        if flag == "HIGH":
            return "High: Elevated level"
        elif flag == "LOW":
            return "Low: Below standard level"
        else:
            return "Normal"

    if flag == "HIGH":
        return "High: Above expected range"
    elif flag == "LOW":
        return "Low: Below expected range"
    elif flag == "NORMAL":
        return "Normal: Within expected range"
    else:
        return "Normal"


# ---------------------------------------------------------------------------
# 3. X-ray image upload & model inference (In-Memory Only)
# ---------------------------------------------------------------------------

@api.route("/analyse", methods=["POST"])
@api.route("/xray/upload", methods=["POST"])
@api.route("/xray/predict", methods=["POST"])
def xray_infer():
    """
    Accepts a chest X-ray image upload (PNG/JPG/JPEG <= 10 MB) or JSON image_path.
    Validates file format, enforces size limits, and checks content integrity.
    Processes image entirely in memory (RAM) and NEVER saves images to disk.
    Does NOT log filenames or image contents.

    Returns the model's class and score/threshold only, with a clear explanation
    that it is experimental and not diagnostic.
    """
    project_root = current_app.root_path
    weights_path = os.path.join(project_root, "artifacts", "tb_xray_baseline.pt")

    # 1. Model Availability Check
    if not os.path.isfile(weights_path):
        return jsonify({
            "error": (
                "Model unavailable: baseline weights file not found at artifacts/tb_xray_baseline.pt. "
                "Please run training or provide model weights."
            ),
            "model_available": False,
            "disclaimer": _DISCLAIMER,
        }), 503

    img_bytes: bytes | None = None

    # 2. Extract input from multipart upload OR json
    if request.files:
        f = request.files.get("file") or request.files.get("image")
        if not f or f.filename == "":
            return jsonify({"error": "No image file provided in 'file' or 'image' field."}), 400

        ext = _ext(f.filename)
        if ext not in ALLOWED_IMG_EXT:
            return jsonify({"error": f"Unsupported file type '{ext}'. Accepted: PNG, JPG, JPEG"}), 415

        img_bytes = f.read()

        if len(img_bytes) == 0:
            return jsonify({"error": "Uploaded image file is empty (0 bytes)."}), 400

        if len(img_bytes) > MAX_IMG_BYTES:
            return jsonify({"error": "Image file too large (max 10 MB)."}), 413

        # Validate content using PIL
        try:
            from PIL import Image
            img_check = Image.open(io.BytesIO(img_bytes))
            img_check.verify()
        except Exception as exc:
            return jsonify({"error": f"Invalid or unreadable image file: {exc}"}), 400

    elif request.is_json:
        data = request.get_json() or {}
        img_path = data.get("image_path")
        if not img_path:
            return jsonify({"error": "Missing 'image_path' in JSON request."}), 400

        full_path = os.path.normpath(os.path.join(project_root, img_path))
        if not full_path.startswith(os.path.abspath(project_root)):
            return jsonify({"error": "Path traversal forbidden."}), 403
        if not os.path.isfile(full_path):
            return jsonify({"error": f"Image file not found: {img_path}"}), 404

        with open(full_path, "rb") as fp:
            img_bytes = fp.read()

    else:
        return jsonify({"error": "No image file or JSON body provided in request."}), 400

    # 2.5 Validation of chest X-ray optical and radiographic characteristics
    is_valid_cxr, cxr_err = validate_chest_xray(img_bytes)
    if not is_valid_cxr:
        return jsonify({
            "error": cxr_err,
            "is_chest_xray": False,
            "status": "invalid_image",
            "disclaimer": _DISCLAIMER,
        }), 400

    # 3. In-memory model inference (never written to disk)
    try:
        pred = predict_chest_xray(img_bytes, project_root=project_root)
        if pred.get("error"):
            return jsonify({
                "error": pred["error"],
                "is_chest_xray": False,
                "status": "invalid_image",
                "disclaimer": _DISCLAIMER,
            }), 400

        return jsonify({
            "model_available":    True,
            "predicted_class":    pred["predicted_label"],
            "predicted_label":    pred["predicted_label"],
            "experimental_score": pred["experimental_score"],
            "threshold":          pred["threshold"],
            "probabilities": {
                "Normal":                 pred["prob_normal"],
                "TB-consistent findings": pred["prob_tb"],
            },
            "status_category":    pred["status_category"],
            "status_color":       pred["status_color"],
            "explanation": (
                "Experimental TB screening score from research baseline only. "
                "This score reflects visual resemblance to documented dataset labels only and "
                "is NOT a medical diagnosis. Consult a qualified physician or radiologist."
            ),
            "is_diagnostic":      False,
            "is_diagnosis":       False,
            "disclaimer":         pred["disclaimer"],
            "limitations":        pred["limitations"],
            "size_bytes":         len(img_bytes),
        }), 200

    except Exception as exc:
        return jsonify({"error": f"Model inference error: {exc}"}), 500


@api.route("/xray/model-info", methods=["GET"])
def xray_model_info():
    """
    Returns architecture, hyperparameters, and held-out test evaluation metrics
    from artifacts/test_metrics.json and artifacts/training_config.json.
    """
    project_root = current_app.root_path
    artifacts_dir = os.path.join(project_root, "artifacts")

    weights_path = os.path.join(artifacts_dir, "tb_xray_baseline.pt")
    metrics_path = os.path.join(artifacts_dir, "test_metrics.json")
    config_path = os.path.join(artifacts_dir, "training_config.json")
    manifest_path = os.path.join(artifacts_dir, "split_manifest.json")

    has_weights = os.path.isfile(weights_path)
    test_metrics = None
    config = None
    manifest_counts = None

    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as fp:
                test_metrics = json.load(fp)
        except Exception:
            pass

    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                config = json.load(fp)
        except Exception:
            pass

    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fp:
                m = json.load(fp)
                manifest_counts = m.get("counts")
        except Exception:
            pass

    return jsonify({
        "model_loaded": has_weights,
        "architecture": config.get("architecture", "resnet18") if config else "resnet18",
        "input_resolution": config.get("input_resolution", [224, 224]) if config else [224, 224],
        "seed": config.get("seed", 42) if config else 42,
        "split_counts": manifest_counts,
        "training_history": config.get("history", []) if config else [],
        "test_metrics": test_metrics,
        "patient_grouping_limitation": PATIENT_GROUPING_LIMITATION,
        "clinical_disclaimer": CLINICAL_DISCLAIMER,
        "limitations": MODEL_LIMITATIONS,
    }), 200


@api.route("/xray/sample-images", methods=["GET"])
def xray_sample_images():
    """
    Returns a curated list of sample images from the local inventory for
    quick one-click evaluation in the frontend demo.
    """
    project_root = current_app.root_path
    db_base = os.path.join(project_root, "data", "raw", "TB_Chest_Radiography_Database")

    normal_dir = os.path.join(db_base, "Normal")
    tb_dir = os.path.join(db_base, "Tuberculosis")

    samples: list[dict[str, str]] = []

    # Select 3 Normal samples
    if os.path.isdir(normal_dir):
        normals = sorted(os.listdir(normal_dir))[:3]
        for name in normals:
            rel = os.path.join("data", "raw", "TB_Chest_Radiography_Database", "Normal", name).replace("\\", "/")
            samples.append({
                "filename": name,
                "label_name": "Normal",
                "folder_label": "Normal",
                "path": rel,
            })

    # Select 3 TB samples
    if os.path.isdir(tb_dir):
        tbs = sorted(os.listdir(tb_dir))[:3]
        for name in tbs:
            rel = os.path.join("data", "raw", "TB_Chest_Radiography_Database", "Tuberculosis", name).replace("\\", "/")
            samples.append({
                "filename": name,
                "label_name": "TB-consistent findings",
                "folder_label": "Tuberculosis",
                "path": rel,
            })

    return jsonify({"samples": samples}), 200


@api.route("/xray/image", methods=["GET"])
def xray_serve_image():
    """
    Serve a raw image from the TB dataset safely.
    Query param: path (relative to project root).
    """
    project_root = current_app.root_path
    rel_path = request.args.get("path", "")
    if not rel_path:
        return jsonify({"error": "Missing 'path' parameter"}), 400

    full_path = os.path.normpath(os.path.join(project_root, rel_path))
    # Security: ensure path is within data/raw/TB_Chest_Radiography_Database
    allowed_base = os.path.abspath(os.path.join(project_root, "data", "raw", "TB_Chest_Radiography_Database"))
    if not os.path.abspath(full_path).startswith(allowed_base):
        return jsonify({"error": "Access denied"}), 403

    if not os.path.isfile(full_path):
        return jsonify({"error": "Image not found"}), 404

    return send_file(full_path, mimetype="image/png")


# ---------------------------------------------------------------------------
# 5. X-ray inventory
# ---------------------------------------------------------------------------

@api.route("/xray/inventory", methods=["GET"])
def xray_inventory():
    """
    Scans data/raw/TB_Chest_Radiography_Database/ and returns image counts.
    Returns a clear error message if the dataset is missing or malformed.
    """
    project_root = current_app.root_path
    inv = run_inventory(project_root)
    status_code = 200 if inv.get("db_exists") and not inv.get("error") else 200
    return jsonify(inv), status_code


# ---------------------------------------------------------------------------
# 5. Dataset status (legacy + new combined)
# ---------------------------------------------------------------------------

@api.route("/dataset/status", methods=["GET"])
def dataset_status():
    """
    Returns status for both the TB X-ray database and the lab CSV datasets.
    """
    project_root = current_app.root_path

    # X-ray inventory
    inv = run_inventory(project_root)

    # Lab datasets
    raw_dir = os.path.join(project_root, "data", "raw")
    blood_csv = os.path.join(raw_dir, "Blood_Test_Dataset.csv")
    lab_csv   = os.path.join(raw_dir, "laboratory__data.csv")
    ref_pdf   = os.path.join(raw_dir, "Blood Test Reference Ranges.pdf")

    def _csv_count(path: str) -> int | str:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return sum(1 for _ in csv.reader(f)) - 1   # subtract header
        except Exception as e:
            return f"Error: {e}"

    return jsonify({
        "xray_database":   inv,
        "blood_test_csv":  {
            "path":   blood_csv,
            "exists": os.path.isfile(blood_csv),
            "rows":   _csv_count(blood_csv) if os.path.isfile(blood_csv) else 0,
        },
        "laboratory_csv":  {
            "path":   lab_csv,
            "exists": os.path.isfile(lab_csv),
            "rows":   _csv_count(lab_csv) if os.path.isfile(lab_csv) else 0,
        },
        "reference_pdf":   {
            "path":   ref_pdf,
            "exists": os.path.isfile(ref_pdf),
        },
        "disclaimer": _DISCLAIMER,
    })


# ---------------------------------------------------------------------------
# 6. Blood-test dataset comparison (population statistics)
# ---------------------------------------------------------------------------

@api.route("/lab/dataset-stats", methods=["GET"])
def lab_dataset_stats():
    """
    Returns aggregate statistics from the bundled Blood_Test_Dataset.csv.
    Used by the UI to show population context (not patient diagnosis).
    """
    project_root = current_app.root_path
    csv_path = os.path.join(project_root, "data", "raw", "Blood_Test_Dataset.csv")

    if not os.path.isfile(csv_path):
        return jsonify({"error": "Blood_Test_Dataset.csv not found.", "stats": {}}), 404

    try:
        stats: dict[str, dict] = {}
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            columns: dict[str, list[float]] = {}
            healthy_columns: dict[str, list[float]] = {}
            total = 0
            healthy_count = 0

            for row in reader:
                total += 1
                flag = str(row.get("Health_Flag", "")).strip()
                is_healthy = flag == "1"
                if is_healthy:
                    healthy_count += 1

                for col, val in row.items():
                    if col in ("Health_Flag", "Gender"):
                        continue
                    try:
                        num = float(val)
                        columns.setdefault(col, []).append(num)
                        if is_healthy:
                            healthy_columns.setdefault(col, []).append(num)
                    except ValueError:
                        pass

        for col, vals in columns.items():
            if not vals:
                continue
            h_vals = healthy_columns.get(col, [])
            stats[col] = {
                "count":        len(vals),
                "mean":         round(sum(vals) / len(vals), 3),
                "min":          round(min(vals), 3),
                "max":          round(max(vals), 3),
                "healthy_mean": round(sum(h_vals) / len(h_vals), 3) if h_vals else None,
            }

        return jsonify({
            "total_records":   total,
            "healthy_records": healthy_count,
            "stats":           stats,
            "note": (
                "These are population-level aggregate statistics from the "
                "Blood_Test_Dataset.csv (health flag: 1=healthy). "
                "They are shown for context only and do NOT diagnose "
                "any individual. Consult a healthcare professional."
            ),
            "disclaimer": _DISCLAIMER,
        })

    except Exception as exc:
        return jsonify({"error": str(exc), "stats": {}}), 500


# ---------------------------------------------------------------------------
# 7. Symptom Separation & Structured Clinical Domain Analysis (Non-Diagnostic)
# ---------------------------------------------------------------------------

_SYMPTOM_DOMAINS: dict[str, list[str]] = {
    "Respiratory / श्वसन": [
        "cough", "dry cough", "productive cough", "sputum", "phlegm", "hemoptysis",
        "shortness of breath", "dyspnea", "breathlessness", "wheezing", "chest congestion",
        "throat pain", "sore throat", "stridor", "coughing",
        "खांसी", "सूखी खांसी", "बलगम", "सांस", "सांस फूलना", "घरघराहट", "गले में दर्द"
    ],
    "Constitutional & Systemic / दैहिक व सामान्य": [
        "fever", "high fever", "chills", "rigors", "night sweats", "sweating",
        "fatigue", "malaise", "tiredness", "weakness", "lethargy", "weight loss",
        "loss of appetite", "anorexia",
        "बुखार", "कंपकंपी", "पसीना", "रात में पसीना", "थकान", "कमजोरी", "वजन घटना", "भूख न लगना"
    ],
    "Cardiovascular & Thoracic / हृदय व वक्षीय": [
        "chest pain", "chest discomfort", "chest pressure", "chest tightness",
        "palpitations", "rapid heartbeat",
        "सीने में दर्द", "छाती में दर्द", "सीने में भारीपन", "धड़कन"
    ],
    "Gastrointestinal & Other / पाचन व अन्य": [
        "nausea", "vomiting", "abdominal pain", "diarrhea", "stomach pain",
        "headache", "body ache", "myalgia", "joint pain",
        "उल्टी", "मतली", "पेट दर्द", "दस्त", "सिरदर्द", "बदन दर्द", "जोड़ों का दर्द"
    ],
}

_RED_FLAGS: list[dict[str, Any]] = [
    {
        "pattern": re.compile(r"\b(hemoptysis|blood in sputum|coughing blood|blood in cough|खून|बलगम में खून)\b", re.IGNORECASE),
        "name": "Coughing Blood / Hemoptysis (बलगम में खून)",
        "urgency": "Urgent Medical Attention Recommended",
        "advice_en": "Coughing up blood or blood-tinged sputum requires immediate evaluation by a physician or pulmonologist.",
        "advice_hi": "खांसी में खून या बलगम में खून आना एक गंभीर संकेत है। कृपया तुरंत डॉक्टर से मिलें।"
    },
    {
        "pattern": re.compile(r"\b(shortness of breath|severe dyspnea|difficulty breathing|breathless|सांस फूलना|सांस लेने में तकलीफ)\b", re.IGNORECASE),
        "name": "Shortness of Breath / Dyspnea (सांस फूलना)",
        "urgency": "Prompt Medical Attention Recommended",
        "advice_en": "Difficulty breathing or persistent shortness of breath should be physically assessed with pulse oximetry and chest auscultation.",
        "advice_hi": "सांस लेने में कठिनाई या सांस फूलने के लिए तुरंत डॉक्टर से ऑक्सीजन स्तर और फेफड़ों की जांच करवाएं।"
    },
    {
        "pattern": re.compile(r"\b(chest pain|chest pressure|acute chest|सीने में दर्द|छाती में दर्द)\b", re.IGNORECASE),
        "name": "Chest Pain or Pressure (सीने में दर्द)",
        "urgency": "Immediate Medical Assessment Required",
        "advice_en": "New, acute, or radiating chest pain requires immediate medical evaluation to exclude cardiopulmonary causes.",
        "advice_hi": "सीने में तेज या नया दर्द होने पर तुरंत आपातकालीन चिकित्सा जांच करवाएं।"
    },
    {
        "pattern": re.compile(r"\b(high fever|persistent fever|fever >|तेज बुखार|लगातार बुखार)\b", re.IGNORECASE),
        "name": "High or Persistent Fever (तेज या लगातार बुखार)",
        "urgency": "Medical Evaluation Recommended",
        "advice_en": "High fever lasting more than 3 days warrants clinical investigation (CBC, blood culture) to identify underlying infection.",
        "advice_hi": "तीन दिनों से अधिक समय तक तेज बुखार रहने पर संक्रमण की पहचान के लिए डॉक्टर से जांच करवाएं।"
    },
]


def _separate_symptoms(raw_items: list[Any] | None, raw_text: str | None) -> list[dict[str, Any]]:
    items: list[str] = []
    if raw_items:
        for it in raw_items:
            if isinstance(it, str) and it.strip():
                items.append(it.strip())
            elif isinstance(it, dict):
                val = it.get("text") or it.get("name")
                if val:
                    items.append(str(val).strip())

    if raw_text and not items:
        parts = re.split(r"[,;\n\r]+|\b(?:and|with|also|along with|plus|और|तथा|साथ ही)\b", raw_text, flags=re.IGNORECASE)
        for p in parts:
            p_clean = p.strip(" \t\n\r.-*")
            if len(p_clean) >= 2:
                items.append(p_clean)

    separated: list[dict[str, Any]] = []
    seen = set()

    for item in items:
        item_lower = item.lower()
        if item_lower in seen:
            continue
        seen.add(item_lower)

        dur_match = re.search(r"\b(\d+\s*(?:days?|weeks?|months?|hours?|दिन|हफ्ते|महीने|घंटे)(?:\s*(?:ago|duration|se|से))?)\b", item, re.IGNORECASE)
        duration = dur_match.group(1).strip() if dur_match else ""

        domain = "General & Other / अन्य"
        for dom, kws in _SYMPTOM_DOMAINS.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", item, re.IGNORECASE) for kw in kws):
                domain = dom
                break

        red_flag_match = None
        for rf in _RED_FLAGS:
            if rf["pattern"].search(item):
                red_flag_match = rf
                break

        separated.append({
            "name": item,
            "text": item,
            "duration": duration,
            "domain": domain,
            "is_red_flag": red_flag_match is not None,
            "urgency": red_flag_match["urgency"] if red_flag_match else "Routine Observation",
            "clinical_advice_en": red_flag_match["advice_en"] if red_flag_match else "Discuss symptom duration with your physician.",
            "clinical_advice_hi": red_flag_match["advice_hi"] if red_flag_match else "डॉक्टर से इस लक्षण की अवधि पर चर्चा करें।",
        })

    return separated


@api.route("/symptoms/analyze", methods=["POST"])
def analyze_symptoms():
    """
    Accepts raw notes text and/or a list of separated symptoms.
    Parses symptoms into discrete entities, groups by physiological domain,
    and identifies clinical red-flag signs requiring urgent physician attention.
    STRICTLY NON-DIAGNOSTIC: Never diagnoses a disease or prescribes medication.
    """
    body = request.get_json(silent=True) or {}
    raw_text = body.get("text") or body.get("notes") or ""
    raw_items = body.get("symptoms") or []

    separated = _separate_symptoms(raw_items, raw_text)

    # Group by domain
    by_domain: dict[str, list[dict]] = {}
    red_flags: list[dict] = []

    for sym in separated:
        dom = sym["domain"]
        by_domain.setdefault(dom, []).append(sym)
        if sym.get("is_red_flag"):
            red_flags.append(sym)

    DOMAIN_ICONS = {
        "Respiratory / श्वसन": "🫁",
        "Constitutional & Systemic / दैहिक व सामान्य": "🌡️",
        "Cardiovascular & Thoracic / हृदय व वक्षीय": "❤️",
        "Gastrointestinal & Other / पाचन व अन्य": "🩺",
        "General & Other / अन्य": "🩺",
    }

    domains_list = [
        {
            "name": dom,
            "icon": DOMAIN_ICONS.get(dom, "🩺"),
            "symptoms": syms,
        }
        for dom, syms in by_domain.items()
    ]

    # Generate clinical consultation talking points
    discussion_points: list[str] = []
    if any("Respiratory" in dom for dom in by_domain):
        discussion_points.append("Ask doctor whether a physical chest auscultation, spirometry, or sputum test is advised.")
    if any("Constitutional" in dom for dom in by_domain):
        discussion_points.append("Discuss fever progression, temperature records, and fatigue timeline.")
    if any("Cardiovascular" in dom for dom in by_domain):
        discussion_points.append("Discuss chest discomfort triggers, exertional symptoms, and resting pulse.")
    if red_flags:
        discussion_points.insert(0, "⚠️ PRIORITY: Highlight reported red-flag symptoms immediately during consultation.")

    return jsonify({
        "symptoms_count": len(separated),
        "separated_symptoms": separated,
        "domains": domains_list,
        "by_domain": by_domain,
        "red_flags": [rf["name"] for rf in red_flags],
        "red_flag_details": red_flags,
        "has_red_flags": len(red_flags) > 0,
        "discussion_points": discussion_points,
        "disclaimer": "Educational symptom breakdown only. Not a medical diagnosis or treatment recommendation. Always consult a licensed physician.",
    }), 200


# ---------------------------------------------------------------------------
# 8. Multi-Lingual Summary Voice Script Generator (English & Hindi)
# ---------------------------------------------------------------------------

@api.route("/summary/voice-script", methods=["POST"])
def summary_voice_script():
    """
    Generates natural, spoken walkthrough scripts in both English and Hindi
    for the findings summary report.
    Used by the Voice Assistant on Section 4.
    """
    body = request.get_json(silent=True) or {}
    lab_rows = body.get("lab_rows") or []
    xray = body.get("xray_result") or {}
    symptoms = body.get("symptoms") or []
    notes = body.get("notes") or ""

    # 1. English Script
    en_parts = [
        "Welcome to your summary report audio overview.",
    ]

    # Lab section
    if lab_rows:
        flagged = [r for r in lab_rows if (r.get("flag") or "").upper() in ("HIGH", "LOW")]
        en_parts.append(f"In your laboratory report, {len(lab_rows)} test results were extracted.")
        if flagged:
            flag_desc = ", ".join(f"{r.get('test_name', 'Test')} is {r.get('value', '')} {r.get('unit', '')}, marked {r.get('flag', '').lower()}" for r in flagged[:3])
            en_parts.append(f"There are {len(flagged)} out-of-range values, including: {flag_desc}. Remember that a result outside a reference range does not by itself establish a diagnosis.")
        else:
            en_parts.append("All extracted laboratory values are within their respective printed reference intervals.")
    else:
        en_parts.append("No laboratory report was analyzed in this session.")

    # X-ray section
    if xray and not xray.get("error"):
        pred = xray.get("prediction") or xray
        p_class = pred.get("predicted_class") or pred.get("predicted_label") or "Normal"
        score = pred.get("experimental_score")
        score_val = f"{score:.2f}" if score is not None else "0.0"
        en_parts.append(f"For your chest X-ray, the automated baseline screening model categorized the scan as {p_class}, with a screening confidence score of {score_val} against a baseline threshold of 0.50. This is an automated screening score, not a medical diagnosis.")
    else:
        en_parts.append("No chest X-ray was evaluated in this session.")

    # Symptoms section
    if symptoms or notes:
        sym_names = [s.get("name") or s.get("text") or str(s) if isinstance(s, dict) else str(s) for s in symptoms]
        if not sym_names and notes:
            sym_names = [notes[:120]]
        en_parts.append(f"Recorded symptoms include: {', '.join(sym_names[:5])}. These symptoms are direct user observations and were not diagnosed by artificial intelligence.")
    else:
        en_parts.append("No personal symptom observations were recorded.")

    en_parts.append("This research demo summary is for informational guidance only. All findings should be reviewed with a qualified healthcare professional. Thank you.")
    english_script = " ".join(en_parts)

    # 2. Hindi Script (हिंदी सारांश पटकथा)
    hi_parts = [
        "आपके सारांश रिपोर्ट ऑडियो में आपका स्वागत है।",
    ]

    # Lab section Hindi
    if lab_rows:
        flagged = [r for r in lab_rows if (r.get("flag") or "").upper() in ("HIGH", "LOW")]
        hi_parts.append(f"आपकी लैब रिपोर्ट में {len(lab_rows)} जांच मान निकाले गए हैं।")
        if flagged:
            hi_parts.append(f"इनमें से {len(flagged)} मान संदर्भ सीमा से बाहर हैं। कृपया ध्यान दें कि किसी मान का सामान्य सीमा से बाहर होना अपने आप में किसी बीमारी का निदान नहीं है।")
        else:
            hi_parts.append("सभी निकाले गए लैब मान सामान्य संदर्भ सीमा के भीतर हैं।")
    else:
        hi_parts.append("इस सत्र में कोई लैब रिपोर्ट अपलोड नहीं की गई।")

    # X-ray section Hindi
    if xray and not xray.get("error"):
        pred = xray.get("prediction") or xray
        p_class = pred.get("predicted_class") or pred.get("predicted_label") or "Normal"
        score = pred.get("experimental_score")
        score_val = f"{score:.2f}" if score is not None else "0.0"
        p_class_hi = "सामान्य" if "normal" in str(p_class).lower() else "टीबी के संभावित लक्षण"
        hi_parts.append(f"चेस्ट एक्स-रे के लिए, प्रयोगात्मक स्क्रीनिंग मॉडल ने {p_class_hi} का अनुमान लगाया है, जिसका स्क्रीनिंग स्कोर {score_val} है, 0.50 थ्रेशोल्ड के मुकाबले। यह एक प्रयोगात्मक स्कोर है, कोई डॉक्टर का निदान नहीं है।")
    else:
        hi_parts.append("इस सत्र में कोई चेस्ट एक्स-रे इमेज मूल्यांकित नहीं की गई।")

    # Symptoms section Hindi
    if symptoms or notes:
        sym_names = [s.get("name") or s.get("text") or str(s) if isinstance(s, dict) else str(s) for s in symptoms]
        if not sym_names and notes:
            sym_names = [notes[:120]]
        hi_parts.append(f"दर्ज किए गए लक्षणों में शामिल हैं: {', '.join(sym_names[:5])}। ये लक्षण उपयोगकर्ता द्वारा सीधे दर्ज किए गए हैं और किसी एआई द्वारा रोग निदान नहीं किए गए हैं।")
    else:
        hi_parts.append("कोई व्यक्तिगत लक्षण दर्ज नहीं किया गया।")

    hi_parts.append("कृपया अपने सभी निष्कर्षों और लक्षणों पर एक योग्य और लाइसेंस प्राप्त चिकित्सक से परामर्श करें। धन्यवाद।")
    hindi_script = " ".join(hi_parts)

    return jsonify({
        "english_script": english_script,
        "hindi_script": hindi_script,
        "disclaimer": _DISCLAIMER,
    }), 200


# ---------------------------------------------------------------------------
# 9. Smart Hybrid Chatbot Endpoint
# ---------------------------------------------------------------------------

@api.route("/chat", methods=["POST"])
def chat():
    """
    Smart Hybrid Chatbot Endpoint.
    Accepts:
      {
        "message": "User query",
        "history": [{"role": "user", "content": "..."}, ...],
        "session_data": {
          "lab_report": {...},
          "xray": {...},
          "symptoms": {...}
        }
      }
    Returns:
      {
        "reply": str,
        "mode": "cloud_llm" | "clinical_engine",
        "model": str,
        "status": "ok"
      }
    """
    body = request.get_json(silent=True) or {}
    message = body.get("message") or ""
    history = body.get("history") or []
    session_data = body.get("session_data") or {}

    reply, mode, model = get_chatbot_response(message, history, session_data)

    return jsonify({
        "reply": reply,
        "mode": mode,
        "model": model,
        "status": "ok",
    }), 200

