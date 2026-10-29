"""
tests/test_demo.py
Automated smoke-tests for Healthcare Diagnostic Assistant & X-Ray Baseline.
Run with: python -m pytest tests/ -v or python tests/test_demo.py
"""

import io
import json
import os
import struct
import sys
import zlib

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import create_app


def _client():
    """Create a Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# ── Health endpoint ────────────────────────────────────────

def test_health_returns_200():
    client = _client()
    res = client.get("/api/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = json.loads(res.data)
    assert data["status"] == "ok"
    assert "disclaimer" in data
    print("[PASS] GET /api/health -> 200 OK")


def test_health_contains_disclaimer():
    client = _client()
    data = json.loads(client.get("/api/health").data)
    assert "Not a medical device" in data["disclaimer"]
    print("[PASS] Disclaimer text present in /api/health")


# ── Analyse / Upload endpoint ──────────────────────────────

def test_analyse_no_file_returns_400():
    client = _client()
    res = client.post("/api/analyse")
    assert res.status_code == 400
    print("[PASS] POST /api/analyse (no file) -> 400")


def test_analyse_bad_extension_returns_415():
    client = _client()
    data = {"image": (io.BytesIO(b"fake content"), "report.pdf")}
    res = client.post(
        "/api/analyse",
        data=data,
        content_type="multipart/form-data",
    )
    assert res.status_code == 415
    print("[PASS] POST /api/analyse (.pdf) -> 415")


def test_analyse_valid_png_returns_200():
    def _minimal_png() -> bytes:
        sig = b"\x89PNG\r\n\x1a\n"
        def chunk(tag, d):
            c = struct.pack(">I", len(d)) + tag + d
            return c + struct.pack(">I", zlib.crc32(tag + d) & 0xFFFFFFFF)
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw  = b"\x00\xFF\xFF\xFF"
        idat = chunk(b"IDAT", zlib.compress(raw))
        iend = chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    client = _client()
    png = _minimal_png()
    data = {"file": (io.BytesIO(png), "test.png")}
    res = client.post(
        "/api/xray/upload",
        data=data,
        content_type="multipart/form-data",
    )
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.data}"
    body = json.loads(res.data)
    assert "disclaimer" in body
    print("[PASS] POST /api/xray/upload (valid PNG) -> 200")


# ── Dataset status endpoint ────────────────────────────────

def test_dataset_status_returns_200():
    client = _client()
    res = client.get("/api/dataset/status")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "xray_database" in data
    assert data["xray_database"]["db_exists"] is True
    print(f"[PASS] GET /api/dataset/status -> 200 (xray images={data['xray_database']['total_count']})")


# ── Frontend served ────────────────────────────────────────

def test_root_serves_html():
    client = _client()
    res = client.get("/")
    assert res.status_code == 200
    assert b"<!DOCTYPE" in res.data or b"Healthcare" in res.data
    print("[PASS] GET / -> 200 (frontend HTML served)")


# ── X-Ray Baseline Transfer-Learning Tests ─────────────────

def test_xray_model_info_returns_metrics_and_disclaimer():
    client = _client()
    res = client.get("/api/xray/model-info")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = json.loads(res.data)
    assert data["model_loaded"] is True
    assert data["architecture"] == "resnet18"
    assert "test_metrics" in data and data["test_metrics"] is not None

    metrics = data["test_metrics"]
    assert metrics["sample_count"] == 630
    assert metrics["normal_count"] == 525
    assert metrics["tb_count"] == 105
    assert metrics["accuracy"] >= 0.90
    assert metrics["sensitivity"] >= 0.90
    assert metrics["specificity"] >= 0.90
    assert metrics["precision"] >= 0.90
    assert metrics["f1"] >= 0.90
    assert metrics["roc_auc"] >= 0.95
    assert "confusion_matrix" in metrics

    cm = metrics["confusion_matrix"]
    assert cm["tn"] + cm["fp"] == 525
    assert cm["fn"] + cm["tp"] == 105

    # Regulatory & limitation requirements
    assert "clinical_disclaimer" in data
    assert "NOT a medical diagnosis" in data["clinical_disclaimer"]
    assert "patient_grouping_limitation" in data
    assert "Patient grouping cannot be established" in data["patient_grouping_limitation"]
    print("[PASS] GET /api/xray/model-info -> 200 (held-out test metrics & limitations verified)")


def test_xray_sample_images_endpoint():
    client = _client()
    res = client.get("/api/xray/sample-images")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "samples" in data and len(data["samples"]) > 0
    labels = {s["label_name"] for s in data["samples"]}
    assert "Normal" in labels
    assert "TB-consistent findings" in labels
    print("[PASS] GET /api/xray/sample-images -> 200 (curated samples available)")


def test_xray_serve_image():
    client = _client()
    res = client.get("/api/xray/image?path=data/raw/TB_Chest_Radiography_Database/Normal/Normal-1.png")
    assert res.status_code == 200
    assert res.content_type.startswith("image/")
    print("[PASS] GET /api/xray/image -> 200 (sample image served)")


def test_xray_predict_with_sample_image():
    client = _client()
    payload = {"image_path": "data/raw/TB_Chest_Radiography_Database/Normal/Normal-1.png"}
    res = client.post(
        "/api/xray/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.data}"
    data = json.loads(res.data)
    assert "experimental_score" in data
    assert "predicted_label" in data
    assert data["predicted_label"] == "Normal"
    assert data["experimental_score"] < 0.50
    assert "disclaimer" in data
    assert "NOT a medical diagnosis" in data["disclaimer"]
    assert data["is_diagnosis"] is False
    print(f"[PASS] POST /api/xray/predict (Normal-1) -> score={data['experimental_score']}, label={data['predicted_label']}")


def test_xray_predict_tb_sample():
    client = _client()
    payload = {"image_path": "data/raw/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-1.png"}
    res = client.post(
        "/api/xray/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "experimental_score" in data
    assert data["predicted_label"] == "TB-consistent findings"
    assert data["experimental_score"] >= 0.50
    assert data["is_diagnosis"] is False
    print(f"[PASS] POST /api/xray/predict (Tuberculosis-1) -> score={data['experimental_score']}, label={data['predicted_label']}")


def test_model_status_endpoint():
    client = _client()
    res = client.get("/api/model/status")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["model_available"] is True
    assert data["model_architecture"] == "resnet18"
    assert data["threshold"] == 0.50
    assert "classes" in data
    assert "disclaimer" in data
    print("[PASS] GET /api/model/status -> 200 (model available & threshold=0.50)")


def test_lab_parse_empty_file_returns_400():
    client = _client()
    data = {"file": (io.BytesIO(b""), "empty.pdf")}
    res = client.post("/api/lab/parse", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    body = json.loads(res.data)
    assert "error" in body
    print("[PASS] POST /api/lab/parse (empty file) -> 400")


def test_lab_parse_corrupted_pdf_returns_400():
    client = _client()
    data = {"file": (io.BytesIO(b"Not A PDF Header At All"), "fake.pdf")}
    res = client.post("/api/lab/parse", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    body = json.loads(res.data)
    assert "Invalid PDF" in body.get("error", "")
    print("[PASS] POST /api/lab/parse (corrupted PDF header) -> 400")


def test_xray_upload_corrupted_image_returns_400():
    client = _client()
    data = {"file": (io.BytesIO(b"Not A Valid PNG File"), "corrupt.png")}
    res = client.post("/api/xray/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    body = json.loads(res.data)
    assert "Invalid or unreadable image" in body.get("error", "")
    print("[PASS] POST /api/xray/upload (corrupted image bytes) -> 400")


def test_xray_upload_non_cxr_image_rejected_with_400():
    """Verify that uploading a natural color image (non-chest-xray) returns HTTP 400 and clear error."""
    from PIL import Image, ImageDraw
    client = _client()
    im = Image.new("RGB", (200, 200), (255, 50, 50))
    draw = ImageDraw.Draw(im)
    draw.rectangle([(40, 40), (160, 160)], fill=(50, 255, 50))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)

    data = {"file": (buf, "non_cxr_photo.png")}
    res = client.post("/api/xray/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400, f"Expected 400 for non-CXR image, got {res.status_code}: {res.data}"
    body = json.loads(res.data)
    assert body.get("is_chest_xray") is False
    assert "chest X-ray" in body.get("error", "") or "CXR" in body.get("error", "") or "Inappropriate" in body.get("error", "")
    print("[PASS] POST /api/xray/upload (non-CXR image rejected) -> 400")


def test_xray_upload_class_score_threshold_structure():
    def _minimal_png() -> bytes:
        sig = b"\x89PNG\r\n\x1a\n"
        def chunk(tag, d):
            c = struct.pack(">I", len(d)) + tag + d
            return c + struct.pack(">I", zlib.crc32(tag + d) & 0xFFFFFFFF)
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw  = b"\x00\xFF\xFF\xFF"
        idat = chunk(b"IDAT", zlib.compress(raw))
        iend = chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    client = _client()
    data = {"file": (io.BytesIO(_minimal_png()), "xray.png")}
    res = client.post("/api/xray/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    body = json.loads(res.data)
    assert "predicted_class" in body
    assert "experimental_score" in body
    assert body["threshold"] == 0.50
    assert "explanation" in body
    assert "NOT a medical diagnosis" in body["explanation"]
    assert body["is_diagnostic"] is False
    assert body["is_diagnosis"] is False
    print("[PASS] POST /api/xray/upload (model class, score/threshold, non-diagnostic explanation verified)")


def test_lab_parse_valid_pdf_in_memory():
    client = _client()
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "Blood Test Reference Ranges.pdf")
    if os.path.isfile(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        data = {"file": (io.BytesIO(pdf_bytes), "Blood_Test_Reference_Ranges.pdf")}
        res = client.post("/api/lab/parse", data=data, content_type="multipart/form-data")
        assert res.status_code == 200
        body = json.loads(res.data)
        assert "rows" in body
        assert "disclaimer" in body
        print(f"[PASS] POST /api/lab/parse (in-memory parsing of reference PDF) -> 200 (rows={len(body['rows'])})")


def test_lab_parse_scanned_pdf_with_ocr():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (800, 200), color="white")
    d = ImageDraw.Draw(img)
    d.text((30, 80), "Platelet Count   250   10^3/uL   150-450   NORMAL", fill="black")
    pdf_buf = io.BytesIO()
    img.save(pdf_buf, format="PDF")

    client = _client()
    data = {"file": (io.BytesIO(pdf_buf.getvalue()), "scanned_report.pdf")}
    res = client.post("/api/lab/parse", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    body = json.loads(res.data)
    assert body.get("ocr_used") is True
    assert "rows" in body
    print(f"[PASS] POST /api/lab/parse (scanned PDF via Tesseract OCR) -> 200 (ocr_used={body['ocr_used']})")


def test_lab_parse_image_with_ocr():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (600, 100), color="white")
    d = ImageDraw.Draw(img)
    d.text((20, 40), "Hemoglobin   14.5   g/dL   12.0-16.0   Normal", fill="black")
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")

    client = _client()
    data = {"file": (io.BytesIO(img_buf.getvalue()), "report.png")}
    res = client.post("/api/lab/parse", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    body = json.loads(res.data)
    assert body.get("ocr_used") is True
    assert "rows" in body
    print(f"[PASS] POST /api/lab/parse (image report via Tesseract OCR) -> 200 (ocr_used={body['ocr_used']})")


# ── Symptom Recording & Analysis Endpoint ───────────────────

def test_symptom_analyze_endpoint():
    client = _client()
    payload = {
        "notes": "Patient has persistent dry cough for 3 weeks and high fever with shortness of breath",
        "symptoms": [
            {"text": "Persistent dry cough", "duration": "3 weeks"},
            {"text": "Shortness of breath (Dyspnea)"},
            {"text": "High fever"}
        ]
    }
    res = client.post("/api/symptoms/analyze", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = json.loads(res.data)
    assert data["has_red_flags"] is True, "Expected red flags to be detected"
    assert len(data["red_flags"]) >= 1, "Expected at least 1 red flag"
    assert len(data["domains"]) >= 2, "Expected at least 2 domains (Respiratory, Constitutional)"
    assert "discussion_points" in data and len(data["discussion_points"]) > 0
    assert "disclaimer" in data
    print(f"[PASS] POST /api/symptoms/analyze -> 200 (red_flags={data['red_flags']}, domains={len(data['domains'])})")


# ── Bilingual Summary Voice Script Endpoint ─────────────────

def test_voice_script_endpoint():
    client = _client()
    payload = {
        "lab_rows": [
            {"test_name": "Hemoglobin", "value": "14.5", "unit": "g/dL", "flag": "NORMAL"},
            {"test_name": "WBC", "value": "12.8", "unit": "10^3/uL", "flag": "HIGH"}
        ],
        "xray_result": {
            "predicted_class": "Normal",
            "experimental_score": 0.0134,
            "threshold": 0.50
        },
        "symptoms": [
            {"text": "Persistent dry cough", "duration": "2 weeks"}
        ],
        "notes": "Mild fatigue in the evening"
    }
    res = client.post("/api/summary/voice-script", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = json.loads(res.data)
    assert "english_script" in data and len(data["english_script"]) > 50
    assert "hindi_script" in data and len(data["hindi_script"]) > 50
    assert "research" in data["english_script"].lower() or "demo" in data["english_script"].lower()
    print(f"[PASS] POST /api/summary/voice-script -> 200 (EN len={len(data['english_script'])}, HI len={len(data['hindi_script'])})")


# ── Run directly ───────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_health_returns_200,
        test_health_contains_disclaimer,
        test_model_status_endpoint,
        test_analyse_no_file_returns_400,
        test_analyse_bad_extension_returns_415,
        test_analyse_valid_png_returns_200,
        test_lab_parse_empty_file_returns_400,
        test_lab_parse_corrupted_pdf_returns_400,
        test_xray_upload_corrupted_image_returns_400,
        test_xray_upload_non_cxr_image_rejected_with_400,
        test_xray_upload_class_score_threshold_structure,
        test_lab_parse_valid_pdf_in_memory,
        test_lab_parse_scanned_pdf_with_ocr,
        test_lab_parse_image_with_ocr,
        test_dataset_status_returns_200,
        test_root_serves_html,
        test_xray_model_info_returns_metrics_and_disclaimer,
        test_xray_sample_images_endpoint,
        test_xray_serve_image,
        test_xray_predict_with_sample_image,
        test_xray_predict_tb_sample,
        test_symptom_analyze_endpoint,
        test_voice_script_endpoint,
    ]

    passed = failed = 0
    print("\n== Healthcare Diagnostic Assistant - Baseline Test Suite ==\n")
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n================================================")
    print(f"  {passed} passed | {failed} failed")
    print(f"================================================\n")
    sys.exit(failed)
