"""
backend/chatbot.py - Smart Hybrid Medical Chatbot Engine
Combines:
  1. Built-in Clinical Knowledge & Session Data Analyzer (Option C - 100% offline, zero-key, zero-dependency)
  2. Cloud Generative AI via Google Gemini REST API (Option A - automatic upgrade when GEMINI_API_KEY is present)
"""

import os
import json
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Tuple

CLINICAL_DISCLAIMER = ""

# ═══════════════════════════════════════════════════════════════════
# 1. BUILT-IN CLINICAL KNOWLEDGE BASE (Option C)
# ═══════════════════════════════════════════════════════════════════

TB_CONCISE_CONTENT = (
    "### 🔬 What is Tuberculosis (TB)?\n\n"
    "**Tuberculosis (TB)** is an airborne infectious disease caused by the bacterium *Mycobacterium tuberculosis*, primarily affecting the lungs.\n\n"
    "• **How It Spreads**: Through airborne droplets released when an infected person coughs, sneezes, or talks.\n"
    "• **Cardinal Symptoms**: Persistent cough (>3 weeks), fever (often late-afternoon spikes), night sweats, unexplained weight loss, and fatigue.\n"
    "• **Screening & Diagnosis**: Evaluated through chest X-rays, sputum molecular tests (CBNAAT / GeneXpert), and skin/blood tests (Mantoux / IGRA).\n"
    "• **Treatment & Cure**: Curable with a standardized 6-month combination antibiotic regimen taken consistently without interruption.\n\n"
    "---\n"
    "*Would you like further assistance or more details on symptoms, diagnostic testing, or treatment protocols? You can ask below or click the button:*\n\n"
    "[[button:Elaborate on Tuberculosis|🔬 Elaborate on Tuberculosis]]"
)

TB_ELABORATED_CONTENT = (
    "### 🔬 Comprehensive Clinical Guide: Tuberculosis (TB)\n\n"
    "**Tuberculosis (TB)** is an infectious bacterial disease caused by *Mycobacterium tuberculosis*. "
    "While pulmonary TB (in the lungs) is the most frequent form, extrapulmonary TB can develop in the lymph nodes, spine, brain, or abdominal organs.\n\n"
    "#### 1. Detailed Clinical Symptom Profile\n"
    "• **Cough Progression**: Typically begins as a dry cough and progresses to productive phlegm lasting 3 weeks or longer. Cavitary disease can lead to coughing up blood or blood-streaked sputum (hemoptysis).\n"
    "• **Constitutional Symptoms**: Late-afternoon or evening low-grade fever, drenching night sweats requiring clothing changes, loss of appetite (anorexia), and unintentional weight loss.\n"
    "• **Respiratory & Thoracic**: Sharp pleuritic chest pain worsened by deep inhalation or coughing, accompanied by progressive shortness of breath.\n\n"
    "#### 2. Diagnostic & Confirmatory Testing\n"
    "• **CBNAAT / GeneXpert DNA Testing**: Rapid automated molecular assay identifying *M. tuberculosis* DNA from sputum within 2 hours while detecting Rifampicin antibiotic resistance.\n"
    "• **Sputum Smear Microscopy (AFB)**: Acid-Fast Bacilli staining under microscopy to quantify bacterial load and assess contagiousness.\n"
    "• **Chest Radiography (PA View)**: Visualizes hallmark pulmonary features including apical or upper-lobe consolidations, cavitary lesions (hollowed necrotic areas), pleural effusion, or diffuse miliary nodules.\n"
    "• **Immunological Testing**: Mantoux Tuberculin Skin Test (TST) or Interferon-Gamma Release Assays (IGRA blood tests) to detect immune sensitization.\n\n"
    "#### 3. Standard Treatment Protocols\n"
    "• **First-Line Standard Course (DOTS)**: Standard 6-month Directly Observed Therapy utilizing 4 essential antibiotics:\n"
    "  - **Isoniazid (INH)** & **Rifampicin (RIF)**\n"
    "  - **Pyrazinamide (PZA)** & **Ethambutol (EMB)**\n"
    "• **Critical Adherence Notice**: Medications must be taken daily without missing doses to prevent development of Multi-Drug Resistant TB (MDR-TB).\n\n"
    "*Always consult a licensed physician or pulmonologist for medical diagnosis and prescription treatment.*"
)

DOCTOR_CONCISE_CONTENT = (
    "### 🚨 When to See a Doctor\n\n"
    "Timely medical review ensures early intervention and prevents clinical complications.\n\n"
    "• **Persistent Cough**: Any cough lasting 3 weeks or longer, or coughing up any blood (hemoptysis).\n"
    "• **Constitutional Signs**: Unexplained recurring fever, drenching night sweats, or unintentional weight loss.\n"
    "• **Breathing Discomfort**: Shortness of breath during normal daily tasks, wheezing, or chest tightness.\n"
    "• **Immediate Emergency Signs**: Crushing chest pain, severe shortness of breath, bluish lips, or sudden confusion requires immediate emergency care (call 112 / 911 or visit the nearest ER).\n\n"
    "---\n"
    "*Would you like further assistance or more details on emergency warning signs and doctor consultation preparation? You can ask below or click the button:*\n\n"
    "[[button:Elaborate on when to see a doctor|🚨 Elaborate on When to See a Doctor]]"
)

DOCTOR_ELABORATED_CONTENT = (
    "### 🚨 In-Depth Clinical Guide: When to Seek Medical Attention\n\n"
    "Knowing when symptoms require emergency intervention versus scheduled clinic evaluation helps protect your health.\n\n"
    "#### 1. Immediate Emergency Red Flags (Seek Emergency Care / ER Now)\n"
    "Call emergency services (112 / 911) or proceed immediately to the nearest hospital emergency department if you experience:\n"
    "• **Crushing Chest Pain**: Pressure, fullness, or squeezing pain in the center of the chest radiating to the neck, jaw, arm, or back.\n"
    "• **Severe Breathlessness**: Inability to speak in full sentences without gasping, severe air hunger, or audible wheezing/stridor.\n"
    "• **Hemoptysis**: Coughing up significant amounts of fresh red blood.\n"
    "• **Cyanosis**: Bluish or grayish discoloration of the lips, face, or fingernails indicating acute oxygen deficiency.\n"
    "• **Neurological Warning Signs**: Sudden confusion, fainting (syncope), slurred speech, or persistent high fever with neck stiffness.\n\n"
    "#### 2. Urgent Medical Consultation (Within 24–48 Hours)\n"
    "• Continuous high fever (>102°F / 38.9°C) unresponsive to standard fever reducers.\n"
    "• Productive cough with thick yellowish-green purulent sputum accompanied by chills.\n"
    "• Rapidly worsening shortness of breath or sudden lower-extremity swelling.\n\n"
    "#### 3. Scheduled Clinic Visit (Within 1–2 Weeks)\n"
    "• Cough persisting beyond 3 weeks without improvement.\n"
    "• Progressive, unexplained weight loss (>5% body weight within a month or two).\n"
    "• Frequent night sweats that soak your clothing or bed sheets.\n"
    "• Any out-of-range flags on routine laboratory blood tests.\n\n"
    "#### 4. How to Prepare for Your Doctor Appointment\n"
    "• Prepare a concise timeline: when each symptom started, frequency, and changes over time.\n"
    "• Bring your uploaded lab reports, chest radiographs, and a list of all current medications.\n"
    "• Write down 2–3 specific questions for your doctor regarding confirmatory tests or referrals."
)

KNOWLEDGE_TOPICS: Dict[str, Dict[str, Any]] = {
    "cbc": {
        "keywords": ["cbc", "complete blood count", "hemogram", "wbc", "rbc", "platelet", "hemoglobin", "haemoglobin", "esr"],
        "title": "Complete Blood Count (CBC) & Key Reference Ranges",
        "content": (
            "A **Complete Blood Count (CBC)** measures several critical components of blood:\n\n"
            "• **Hemoglobin (Hb)**: Normal adult ~13.0–17.0 g/dL (men), ~12.0–15.5 g/dL (women). Low levels indicate anemia; high levels can occur in dehydration or hypoxia.\n"
            "• **White Blood Cells (WBC / Leukocytes)**: Normal adult ~4,000–11,000 /µL. Elevated counts often indicate active infection, inflammation, or immune stress.\n"
            "• **Platelets (Thrombocytes)**: Normal adult ~150,000–450,000 /µL. Essential for blood clotting. Below 100,000 requires clinical hematology review.\n"
            "• **Erythrocyte Sedimentation Rate (ESR)**: Normal ~0–20 mm/hr. A non-specific indicator of inflammation often elevated in chronic infections like TB or autoimmune disorders."
        ),
    },
    "xray": {
        "keywords": ["consolidation", "infiltrate", "cavity", "resnet18", "cxr radiograph"],
        "title": "Chest Radiography (X-Ray) & AI Screening",
        "content": (
            "**Chest Radiography (CXR)** is a primary non-invasive imaging tool for pulmonary evaluation:\n\n"
            "• **Typical TB Radiographic Features**:\n"
            "  - Apical or upper-lobe consolidations and infiltrates\n"
            "  - Cavitary lung lesions (hollow spaces created by tissue necrosis)\n"
            "  - Pleural effusion (fluid in the pleural space)\n"
            "  - Miliary patterns (multiple small nodules throughout the lungs)\n\n"
            "• **About this System's ResNet18 AI Baseline**:\n"
            "  - This system uses a deep convolutional neural network (ResNet18) to output a binary screening probability against a 0.50 threshold.\n"
            "  - An automated score indicates *visual pattern resemblance* to verified training images and **cannot rule out or confirm active disease**. Professional radiologist review is mandatory."
        ),
    },
    "cough": {
        "keywords": ["cough", "phlegm", "sputum", "bronchitis"],
        "title": "Cough Evaluation & Clinical Red Flags",
        "content": (
            "A **cough** is a protective reflex, but its duration and characteristics guide clinical urgency:\n\n"
            "• **Acute (< 3 weeks)**: Commonly caused by viral upper respiratory infections, common cold, acute bronchitis, or allergies.\n"
            "• **Chronic (≥ 3 weeks)**: Requires formal medical evaluation for TB, asthma, post-nasal drip, GERD, or chronic bronchitis.\n\n"
            "⚠️ **Immediate Red Flags**:\n"
            "  - Coughing up blood or rust-colored sputum (hemoptysis)\n"
            "  - Accompanying shortness of breath or audible stridor\n"
            "  - Inability to complete full sentences without pausing for breath"
        ),
    },
    "emergency": {
        "keywords": ["cyanosis", "dying", "loss of consciousness", "unconscious"],
        "title": "🚨 Emergency Warning Signs (Seek Urgent Care)",
        "content": (
            "🚨 **CRITICAL SAFETY ADVICE**:\n"
            "If you or anyone is experiencing any of the following symptoms, **call local emergency services or proceed immediately to the nearest hospital emergency department**:\n\n"
            "• Severe, crushing chest pain or pressure radiating to the jaw, neck, or arm\n"
            "• Acute shortness of breath or severe difficulty breathing\n"
            "• Coughing up significant amounts of fresh blood\n"
            "• Bluish tint on lips, face, or fingernails (cyanosis)\n"
            "• Sudden confusion, loss of consciousness, or slurred speech\n"
            "• High continuous fever (>103°F / 39.4°C) with neck stiffness or delirium"
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════
# 2. SESSION CONTEXT FORMATTERS & SPECIALIZED QUERY HANDLER
# ═══════════════════════════════════════════════════════════════════

def _clean_query(s: str) -> str:
    return re.sub(r"[?!.,;]+$", "", (s or "").strip().lower()).strip()


def _format_lab_analysis(lab_rows: List[Dict[str, Any]], query: str = "") -> str:
    flagged = [r for r in lab_rows if (r.get("flag") or "").upper() in ("HIGH", "LOW")]
    sections = [
        "### 🧪 Your Uploaded Lab Report Analysis\n",
        f"We identified **{len(lab_rows)} test parameters** in your uploaded report."
    ]

    # If the user queried a specific test (e.g. "platelet", "hemoglobin", "wbc"), highlight it
    q_clean = _clean_query(query)
    if q_clean:
        specific_matches = []
        for r in lab_rows:
            name = (r.get("test_name") or "").lower()
            if any(term in q_clean for term in name.split() if len(term) > 3) or (name and name in q_clean):
                specific_matches.append(r)
        if specific_matches:
            sections.append("\n**Matched Test Query:**")
            for m in specific_matches:
                name = m.get("test_name", "Test")
                val = m.get("value", "")
                unit = m.get("unit", "")
                flag = m.get("flag") or "NORMAL"
                ref = m.get("ref_range") or m.get("reference_interval") or m.get("reference_range", "")
                sections.append(f"- **{name}**: `{val} {unit}` (Status: **{flag}** | Reference: `{ref}`)")

    if flagged:
        sections.append(f"\n⚠️ **{len(flagged)} value(s) fall outside standard reference intervals**:\n")
        for f in flagged:
            name = f.get("test_name", "Test")
            val = f.get("value", "")
            unit = f.get("unit", "")
            flag = f.get("flag", "FLAGGED")
            ref = f.get("ref_range") or f.get("reference_interval") or f.get("reference_range", "")
            sections.append(f"- **{name}**: `{val} {unit}` (Status: **{flag}** | Standard Range: `{ref}`)")
        sections.append(
            "\n*Clinical Context*: An out-of-range value does not automatically mean a disease is present. "
            "Many factors (hydration, recent infection, medication, diurnal variation) can alter lab values. "
            "Discuss these specific numbers with your ordering physician."
        )
    else:
        sections.append("\n✅ All extracted lab values appear **within standard physiological reference intervals**.")
    return "\n".join(sections)


def _format_xray_analysis(xray: Dict[str, Any]) -> str:
    pred_class = xray.get("predicted_class") or xray.get("predicted_label") or "Normal"
    score = xray.get("experimental_score")
    score_str = f"{score:.3f}" if score is not None else "N/A"
    threshold = xray.get("threshold", 0.50)

    sections = [
        "### 🫁 Your Chest X-Ray Model Evaluation\n",
        f"- **Predicted Classification**: **{pred_class}**",
        f"- **Experimental Screening Score**: `{score_str}` (Classification Threshold: `{threshold}`)"
    ]

    if "tb" in pred_class.lower() or (score is not None and score >= threshold):
        sections.append(
            "\n⚠️ **Observation**: The experimental ResNet18 model detected visual pattern features resembling documented TB-positive patterns in the research dataset. "
            "**This is NOT a clinical diagnosis**. It indicates that a radiologist should carefully evaluate the lung fields for apical infiltrates, cavities, or effusions."
        )
    else:
        sections.append(
            "\n✅ **Observation**: The model output is below the 0.50 screening threshold, showing pattern consistency with normal lung radiograms in the dataset. "
            "However, early-stage or extrapulmonary conditions can appear radiographically subtle; full physician review remains essential."
        )
    return "\n".join(sections)


def _format_symptoms_analysis(symptoms: Dict[str, Any]) -> str:
    separated_syms = symptoms.get("separated_symptoms") or []
    notes = symptoms.get("notes") or ""
    red_flags = symptoms.get("red_flags") or []

    sections = ["### 📝 Your Recorded Symptoms\n"]
    if separated_syms:
        sym_names = [s.get("name") if isinstance(s, dict) else str(s) for s in separated_syms]
        sections.append(f"Recorded observations: **{', '.join(sym_names)}**.")
    elif notes:
        sections.append(f"Notes on file: *\"{notes[:200]}\"*")

    if red_flags:
        rf_names = [rf["name"] if isinstance(rf, dict) else str(rf) for rf in red_flags]
        sections.append(f"\n🚨 **Red Flags Identified**: **{', '.join(rf_names)}** — these symptoms warrant prompt medical review.")
    else:
        sections.append("\nNo high-risk emergency red-flag triggers were flagged in the recorded symptoms.")
    return "\n".join(sections)


def _is_lab_results_query(msg: str) -> bool:
    m = _clean_query(msg)
    exact = {
        "explain my lab results", "explain my lab report", "explain lab results",
        "explain my blood results", "explain my test results", "review my lab results",
        "review my lab report", "interpret my lab results", "analyze my lab results",
        "my lab results", "my lab report", "my report", "my test results",
        "what do my lab results mean", "what does my lab report say",
        "what is abnormal on my report", "what is abnormal in my report",
        "what is abnormal in my lab report", "is anything abnormal in my report",
        "is anything abnormal on my report", "are my results normal",
        "show my abnormal values", "what did my report show", "any issues in my report",
        "what is wrong with my report", "check my report", "check my blood test"
    }
    if m in exact:
        return True

    # 1. Report / Test / Lab Subject Words
    report_words = [
        "report", "reports", "lab", "labs", "test", "tests", "blood", "cbc",
        "result", "results", "value", "values", "parameter", "parameters"
    ]
    has_report_subj = any(w in m for w in report_words)

    # 2. Abnormality / Status Words
    abnormal_words = [
        "abnormal", "abnormality", "abnormalities", "high", "low", "out of range",
        "flag", "flags", "flagged", "issue", "issues", "wrong", "bad", "elevated",
        "normal", "okay", "fine", "problem", "problems", "critical"
    ]
    has_abnormal = any(w in m for w in abnormal_words)

    if has_report_subj and has_abnormal:
        return True

    # 3. Action words paired with report targets
    action_words = [
        "explain", "review", "interpret", "analyze", "check", "examine",
        "what is", "what are", "what does", "what do", "what did", "tell me",
        "show", "summarize", "how is", "how are", "can you check", "see", "read", "look"
    ]
    has_action = any(w in m for w in action_words)

    target_words = [
        "my report", "the report", "uploaded report", "my lab", "the lab",
        "my test", "my tests", "the test", "the tests", "my blood", "my results",
        "the results", "my values", "the values", "lab results", "lab report",
        "blood report", "blood results", "blood test", "test report"
    ]
    has_target = any(w in m for w in target_words)

    if has_action and (has_target or (has_report_subj and any(p in m for p in ["my", "the", "this", "uploaded"]))):
        return True

    # 4. Standalone phrases
    if any(p in m for p in [
        "abnormal value", "abnormal result", "abnormal test", "high value", "low value",
        "out of range", "high or low", "flagged value", "flagged result"
    ]):
        return True

    # 5. Queries mentioning specific lab tests with personal or status inquiries
    test_names = [
        "platelet", "platelets", "hemoglobin", "haemoglobin", "wbc", "rbc",
        "glucose", "sugar", "creatinine", "cholesterol", "esr", "urea",
        "bilirubin", "ast", "alt", "protein", "tsh", "calcium", "sodium",
        "potassium", "triglycerides", "lipid", "kft", "lft"
    ]
    if any(t in m for t in test_names) and any(w in m for w in ["my", "the", "level", "levels", "count", "high", "low", "normal", "abnormal", "result", "is", "how"]):
        return True

    return False


def _is_xray_query(msg: str) -> bool:
    m = _clean_query(msg)
    xray_terms = [
        "x-ray", "xray", "chest scan", "chest radiograph", "radiograph",
        "radiography", "cxr", "chest image", "x-ray image", "xray image", "lung scan"
    ]
    has_xray = any(w in m for w in xray_terms)

    # Score or AI model inquiry
    score_terms = [
        "my score", "the score", "x-ray score", "xray score", "screening score",
        "model score", "ai score", "prediction"
    ]
    has_score = any(w in m for w in score_terms)

    if has_xray and has_score:
        return True

    # Action / inquiry on x-ray
    action_words = [
        "what is", "what does", "what did", "what do", "explain", "review",
        "interpret", "analyze", "check", "tell me", "how is", "how's",
        "show", "is my", "does my", "did my", "did the", "find", "mean"
    ]
    has_action = any(w in m for w in action_words)

    if has_xray and (has_action or any(p in m for p in ["normal", "abnormal", "tb", "tuberculosis", "result", "findings", "finding", "clear", "okay"])):
        return True

    if has_score and any(w in m for w in ["what", "mean", "explain", "review", "is", "how"]):
        return True

    return False


def _is_symptoms_query(msg: str) -> bool:
    m = _clean_query(msg)
    sym_terms = [
        "symptom", "symptoms", "complaint", "complaints", "what i typed",
        "what i entered", "recorded symptom", "recorded symptoms", "my notes", "symptom notes"
    ]
    has_sym = any(w in m for w in sym_terms)

    action_words = [
        "review", "check", "analyze", "evaluate", "explain", "what do",
        "what does", "what did", "tell me", "are my", "is my", "look", "how are", "see"
    ]
    has_action = any(w in m for w in action_words)

    risk_words = [
        "red flag", "red flags", "serious", "danger", "dangerous", "bad",
        "worry", "worrisome", "emergency", "urgent"
    ]
    has_risk = any(w in m for w in risk_words)

    if has_sym and (has_action or has_risk or "my" in m):
        return True

    if has_risk and any(w in m for w in ["i have", "feel", "feeling", "typed", "entered"]):
        return True

    return False


def _is_tb_elaborate_query(msg: str) -> bool:
    m = _clean_query(msg)
    has_elaborate = any(w in m for w in [
        "elaborate", "more detail", "more details", "in detail",
        "in-depth", "tell me more", "further details", "deep dive",
        "thoroughly", "detailed"
    ])
    has_tb = any(w in m for w in ["tuberculosis", "tb"])
    return has_elaborate and has_tb


def _is_tb_concise_query(msg: str) -> bool:
    m = _clean_query(msg)
    if _is_tb_elaborate_query(msg):
        return False

    has_tb = any(w in m for w in ["tuberculosis", "tb"])
    tb_questions = [
        "what is", "tell me about", "what causes", "how is", "how does",
        "explain tb", "explain tuberculosis", "about tb", "about tuberculosis",
        "basics", "is tb", "is tuberculosis", "symptoms of tb", "symptoms of tuberculosis",
        "how does tb spread", "tb transmission"
    ]
    return has_tb and any(q in m for q in tb_questions)


def _is_doctor_elaborate_query(msg: str) -> bool:
    m = _clean_query(msg)
    has_elaborate = any(w in m for w in [
        "elaborate", "more detail", "more details", "in detail",
        "in-depth", "tell me more", "further details", "deep dive",
        "thoroughly", "detailed"
    ])
    has_doctor = any(w in m for w in [
        "doctor", "emergency", "when to see", "warning signs", "clinic", "hospital", "urgent"
    ])
    return has_elaborate and has_doctor


def _is_doctor_concise_query(msg: str) -> bool:
    m = _clean_query(msg)
    if _is_doctor_elaborate_query(msg):
        return False

    doctor_phrases = [
        "when to see a doctor", "when to see doctor", "when should i see a doctor",
        "when to go to the doctor", "when to visit a doctor", "when to consult a doctor",
        "when is it an emergency", "emergency warning signs", "do i need a doctor",
        "should i see a doctor", "should i go to the doctor", "should i visit a doctor",
        "when to go to the hospital", "when to go to the er", "warning signs",
        "when to consult", "should i visit clinic"
    ]
    return any(p in m for p in doctor_phrases)


def _is_general_session_query(msg: str) -> bool:
    m = _clean_query(msg)
    exact = {
        "explain my results", "review my data", "review my results",
        "what did you find", "explain my findings", "review my findings",
        "summarize my session", "summarize my findings", "summarize everything",
        "analyze my data", "give me a summary", "overall summary",
        "session summary", "what are my findings", "review everything"
    }
    if m in exact:
        return True

    summary_words = ["summary", "summarize", "overview", "findings", "everything", "all my data", "all results"]
    action_words = ["give", "show", "tell", "what is", "what are", "review", "explain", "provide"]
    return any(s in m for s in summary_words) and any(a in m for a in action_words)


def handle_specialized_query(message: str, session: Dict[str, Any]) -> str | None:
    """
    Handles personal session queries and concise/elaborated medical guides.
    Returns response string if handled, or None to fall back to general Q&A.
    """
    # 1. Elaborate on Tuberculosis
    if _is_tb_elaborate_query(message):
        return TB_ELABORATED_CONTENT

    # 2. Elaborate on When to See a Doctor
    if _is_doctor_elaborate_query(message):
        return DOCTOR_ELABORATED_CONTENT

    # 3. Concise What is Tuberculosis?
    if _is_tb_concise_query(message):
        return TB_CONCISE_CONTENT

    # 4. Concise When to See a Doctor?
    if _is_doctor_concise_query(message):
        return DOCTOR_CONCISE_CONTENT

    # 5. Lab results query
    if _is_lab_results_query(message):
        lab_report = session.get("lab_report") or {}
        lab_rows = lab_report.get("lab_rows") or lab_report.get("rows") or session.get("lab_rows") or []
        if lab_rows:
            return _format_lab_analysis(lab_rows, query=message)
        return (
            "⚠️ **Please upload your lab report data first.**\n\n"
            "No lab report has been uploaded in this session. "
            "Please upload your lab report (PDF or image) in **Lab Report**, "
            "and I will analyze your test values and reference intervals."
        )

    # 6. Chest X-Ray query
    if _is_xray_query(message):
        xray = session.get("xray") or {}
        has_xray = bool(
            xray and (
                xray.get("experimental_score") is not None
                or xray.get("predicted_class")
                or xray.get("predicted_label")
            )
        )
        if has_xray:
            return _format_xray_analysis(xray)
        return (
            "⚠️ **Please upload your chest X-ray first.**\n\n"
            "No chest X-ray has been uploaded or evaluated in this session. "
            "Please upload your radiograph in **Chest X-Ray**, "
            "and I will explain your screening score and model evaluation."
        )

    # 7. Symptoms query
    if _is_symptoms_query(message):
        symptoms = session.get("symptoms") or {}
        has_symptoms = bool(
            symptoms.get("separated_symptoms")
            or (symptoms.get("notes") or "").strip()
        )
        if has_symptoms:
            return _format_symptoms_analysis(symptoms)
        return (
            "⚠️ **Please record your symptoms first.**\n\n"
            "No symptoms have been recorded in this session. "
            "Please enter or speak your symptoms in **Symptom Notes**, "
            "and I will review them and check for any clinical red flags."
        )

    # 8. General session results query
    if _is_general_session_query(message):
        lab_rows = (session.get("lab_report") or {}).get("lab_rows") or (session.get("lab_report") or {}).get("rows") or session.get("lab_rows") or []
        xray = session.get("xray") or {}
        has_xray = bool(xray and (xray.get("experimental_score") is not None or xray.get("predicted_class")))
        symptoms = session.get("symptoms") or {}
        has_symptoms = bool(symptoms.get("separated_symptoms") or (symptoms.get("notes") or "").strip())

        if not lab_rows and not has_xray and not has_symptoms:
            return (
                "⚠️ **Please upload your data first.**\n\n"
                "No diagnostic data or symptoms have been uploaded in this session yet. "
                "Please upload a lab report in **Lab Report**, an X-ray in **Chest X-Ray**, "
                "or record symptoms in **Symptom Notes**, and I will be happy to review them."
            )
        parts = []
        if lab_rows:
            parts.append(_format_lab_analysis(lab_rows, query=message))
        if has_xray:
            parts.append(_format_xray_analysis(xray))
        if has_symptoms:
            parts.append(_format_symptoms_analysis(symptoms))
        return "\n\n---\n\n".join(parts)

    return None


# ═══════════════════════════════════════════════════════════════════
# 3. RULE-BASED RESPONSE GENERATOR (Option C)
# ═══════════════════════════════════════════════════════════════════

def generate_clinical_engine_reply(message: str, session: Dict[str, Any]) -> str:
    """
    Pure Python rule-based and semantic matcher.
    Zero external dependencies, 100% offline and robust.
    """
    msg_clean = message.lower().strip()

    # 1. First, check specialized queries (session queries, concise TB & Doctor, elaborations)
    specialized = handle_specialized_query(message, session)
    if specialized:
        return specialized + CLINICAL_DISCLAIMER

    # 2. Check for emergency keywords
    if any(k in msg_clean for k in KNOWLEDGE_TOPICS["emergency"]["keywords"]):
        return KNOWLEDGE_TOPICS["emergency"]["content"] + CLINICAL_DISCLAIMER

    # 3. Match general medical knowledge topics
    matched_topics: List[str] = []
    for topic_key, topic_data in KNOWLEDGE_TOPICS.items():
        if topic_key == "emergency":
            continue
        if any(re.search(r"\b" + re.escape(kw) + r"\b", msg_clean) for kw in topic_data["keywords"]):
            matched_topics.append(f"### {topic_data['title']}\n{topic_data['content']}")

    if matched_topics:
        return "\n\n---\n\n".join(matched_topics) + CLINICAL_DISCLAIMER

    # 4. Greetings and conversational queries
    greetings = ["hi", "hello", "hey", "namaste", "good morning", "good evening", "good afternoon"]
    if any(msg_clean == g or msg_clean.startswith(g + " ") for g in greetings):
        has_lab = bool(session.get("lab_report", {}).get("lab_rows"))
        has_xray = bool(session.get("xray"))
        has_sym = bool(session.get("symptoms", {}).get("separated_symptoms") or session.get("symptoms", {}).get("notes"))

        status_items = []
        if has_lab:
            status_items.append("🧪 Lab report loaded")
        if has_xray:
            status_items.append("🫁 Chest X-ray evaluated")
        if has_sym:
            status_items.append("📝 Symptoms recorded")

        status_line = f"Active session data: {', '.join(status_items)}." if status_items else "No session data uploaded yet."

        return (
            f"Hello! I am **Vida**, your Medical Screening & Diagnostic Assistant.\n\n"
            f"{status_line}\n\n"
            f"Here are some things you can ask me:\n"
            f"• *\"Explain my lab results\"* (reviews high/low flags)\n"
            f"• *\"What does my X-ray score mean?\"* (explains ResNet18 output)\n"
            f"• *\"Review my symptoms\"* (checks domains & red flags)\n"
            f"• *\"What is Tuberculosis?\"*\n"
            f"• *\"What are normal hemoglobin and platelet levels?\"*\n"
            f"• *\"When to see a doctor?\"*"
            + CLINICAL_DISCLAIMER
        )

    # 5. Default fallback when Vida cannot answer or doesn't know what to say
    return "I can't help you with that for now."


# ═══════════════════════════════════════════════════════════════════
# 4. CLOUD LLM ENGINE (Option A - Google Gemini via REST API)
# ═══════════════════════════════════════════════════════════════════

def _call_gemini_api(api_key: str, message: str, history: List[Dict[str, str]], session: Dict[str, Any]) -> str:
    """
    Calls Google Gemini REST API using Python's built-in urllib (no extra dependencies).
    Supports Gemini 1.5 Flash.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    # Prepare session data context string
    session_context_parts = []
    lab_report = session.get("lab_report") or {}
    lab_rows = lab_report.get("lab_rows") or []
    if lab_rows:
        flagged = [r for r in lab_rows if (r.get("flag") or "").upper() in ("HIGH", "LOW")]
        session_context_parts.append(f"LAB REPORT: {len(lab_rows)} tests extracted. Out-of-range flagged tests: {json.dumps(flagged)}")
    else:
        session_context_parts.append("LAB REPORT: None uploaded in this session.")

    xray = session.get("xray") or {}
    if xray:
        session_context_parts.append(
            f"CHEST X-RAY: Predicted class '{xray.get('predicted_class')}', score {xray.get('experimental_score')} (threshold {xray.get('threshold', 0.5)})."
        )
    else:
        session_context_parts.append("CHEST X-RAY: None evaluated in this session.")

    symptoms = session.get("symptoms") or {}
    notes = symptoms.get("notes") or ""
    sep_syms = symptoms.get("separated_symptoms") or []
    red_flags = symptoms.get("red_flags") or []
    session_context_parts.append(
        f"SYMPTOMS: Raw notes: '{notes}'. Identified symptoms: {json.dumps(sep_syms)}. Red flags: {json.dumps(red_flags)}."
    )

    session_summary = "\n".join(session_context_parts)

    system_instruction = (
        "You are Vida, an empathetic, knowledgeable, and safety-conscious AI Medical Diagnostic Assistant. "
        "You are embedded inside a healthcare screening web application.\n"
        "CRITICAL CLINICAL BOUNDARIES:\n"
        "1. You are an educational screening aid only. Never provide definitive diagnoses or prescribe medications.\n"
        "2. Ground your explanations in the patient's active session data provided below whenever relevant.\n"
        "3. If the patient has out-of-range lab tests or high X-ray screening scores, clearly explain what they signify, suggest questions for their doctor, and reassure them that screening aids are not diagnoses.\n"
        "4. If emergency or red-flag signs appear (chest pain, hemoptysis, severe breathing difficulty), immediately urge urgent in-person emergency care.\n"
        "5. Respond in clean, readable GitHub markdown with bullet points where appropriate.\n"
        "6. Answer in the language the user speaks (fluent in both English and Hindi).\n\n"
        f"CURRENT PATIENT SESSION DATA:\n{session_summary}"
    )

    # Build conversation contents
    contents: List[Dict[str, Any]] = []

    # Insert historical turns (limit to last 6 for prompt efficiency)
    for turn in history[-6:]:
        role = "user" if turn.get("role") == "user" else "model"
        content = turn.get("content", "")
        if content:
            contents.append({
                "role": role,
                "parts": [{"text": content}]
            })

    # Add the current user message
    contents.append({
        "role": "user",
        "parts": [{"text": message}]
    })

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1000,
            "topP": 0.9,
        }
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=12) as response:
        res_body = response.read().decode("utf-8")
        data = json.loads(res_body)

    candidates = data.get("candidates") or []
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            text = parts[0].get("text", "").strip()
            return text + CLINICAL_DISCLAIMER

    raise ValueError("Empty response from Gemini API")


# ═══════════════════════════════════════════════════════════════════
# 5. SMART HYBRID DISPATCHER
# ═══════════════════════════════════════════════════════════════════

def get_chatbot_response(
    message: str,
    history: List[Dict[str, str]] = None,
    session_data: Dict[str, Any] = None
) -> Tuple[str, str, str]:
    """
    Main entry point for chatbot inquiries.
    Returns: (reply_text, mode, model_name)
      mode is 'cloud_llm' or 'clinical_engine'
    """
    history = history or []
    session_data = session_data or {}
    message = (message or "").strip()

    if not message:
        return ("Please enter a medical or diagnostic question.", "clinical_engine", "System")

    # 1. Specialized query handler (Session queries + Concise/Elaborate TB & Doctor)
    specialized = handle_specialized_query(message, session_data)
    if specialized:
        return (specialized, "clinical_engine", "Clinical Knowledge Engine (Offline Ready)")

    # 2. Check for Gemini API key
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if gemini_key and len(gemini_key.strip()) > 10:
        try:
            llm_reply = _call_gemini_api(gemini_key.strip(), message, history, session_data)
            return (llm_reply, "cloud_llm", "Google Gemini 1.5 Flash")
        except Exception as e:
            # Smooth fallback to Option C on any API / network / quota issue
            print(f"[Chatbot] Gemini API unavailable or failed ({e}); switching to clinical engine fallback.")

    # 2. Default: Option C Built-in Clinical Knowledge Engine
    engine_reply = generate_clinical_engine_reply(message, session_data)
    return (engine_reply, "clinical_engine", "Clinical Knowledge Engine (Offline Ready)")
