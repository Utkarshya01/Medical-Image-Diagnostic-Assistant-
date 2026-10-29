# Healthcare Diagnostic Assistant — Experimental TB Screening & Lab Report Review

> ⚠️ **PERSISTENT CLINICAL & REGULATORY SAFETY DISCLAIMER**
>
> This software is a **research and demonstration screening aid only**. It is **NOT a medical device**, has **not** been cleared or evaluated by any health regulatory authority (such as the FDA, EMA, or CDSCO), and must **NOT** be used for clinical diagnosis, patient treatment, triage, or ruling out disease.
>
> - **The chest X-ray score is an experimental screening score, NOT a medical diagnosis.**
> - **A laboratory result outside a reference range does NOT by itself establish a diagnosis.**
> - **This tool cannot diagnose or rule out any condition.**
> - **All diagnostic interpretations must be performed by a qualified, licensed physician or radiologist.**

---

## 1. Project Overview & Architecture

The **Healthcare Diagnostic Assistant** is a local, privacy-first web application designed for exploratory review of clinical lab reports and chest radiography.

### Privacy-First In-Memory Processing Guarantee
- **Local host binding only**: Flask binds strictly to `127.0.0.1:5000` — zero external network requests or telemetry.
- **In-memory file processing**: All uploads (PDFs, images) and user notes are parsed and evaluated directly in system RAM (`io.BytesIO`) and discarded immediately upon response.
- **Zero persistence**: No files, raw text, or images are written to `uploads/` or stored on disk.
- **Zero logging**: Filenames, report text, image contents, and user symptom notes are never written to log files.

> 📖 **Comprehensive Technical Documentation**: For an exhaustive, feature-by-feature breakdown of every technology, deep learning model architecture, training hyperparameters, evaluation metrics, dataset specifications, and implementation details across the entire system, see [TECH_STACK_AND_ARCHITECTURE.md](TECH_STACK_AND_ARCHITECTURE.md).

---

## 2. Four-Section Workflow Overview

The interface features a vertical sidebar with four core workflow sections:

```
[Sidebar Navigation]
  1. 🧪 Lab Report        -> Upload lab-report PDF/image & review extracted values & definitions
  2. 🫁 Chest X-Ray       -> Upload radiograph, test ResNet18 baseline model & view score/threshold
  3. 📝 Symptom Notes     -> Enter optional user notes (strictly unanalyzed by AI)
  4. 📋 Findings Summary  -> Combined overview with each data source clearly labeled & separated
```

### Section 1: Lab Report (Upload & Review)
- Accepts single PDF or image (`PNG`, `JPG`, `JPEG`) up to 20 MB.
- Validates file headers (`%PDF-` header check, PIL structure check) in memory.
- In-memory text extraction using `pdfplumber` with automatic OCR fallback via `pytesseract`.
- Structured table rendering: test names, numeric values, units, printed reference ranges, and flags (`HIGH`, `LOW`, `NORMAL`).
- Plain-language explanations based **strictly on the report's own printed ranges**.
- **Plain-language definitions card** explaining:
  - 🧪 **Test Value**: The specific numerical concentration or qualitative finding detected in your specimen.
  - 📏 **Unit of Measurement**: The standardized metric scale used to express concentration (e.g. `g/dL`, `mg/dL`, `10³/µL`).
  - 📊 **Reference Range**: The numerical interval observed in ~95% of a healthy population tested under standardized conditions.
- **Critical Clinical Reminder**: A result outside a reference range does *not* by itself establish a diagnosis or mean an illness is present. Transient physiological variations (hydration, stress, diet, activity) frequently cause values outside normal limits.

### Section 2: Chest X-Ray (Upload & Baseline Model Output)
- Accepts chest X-ray image (`PNG`, `JPG`, `JPEG`) up to 10 MB.
- Pretrained `torchvision` ResNet18 transfer-learning baseline with binary linear head (Normal vs TB-consistent findings).
- Displays the experimental screening score (0.0000 to 1.0000), decision threshold (`0.50`), predicted class, and probability gauge bar.
- Interactive sample testing with verified dataset images (`Normal-1`, `Tuberculosis-1`, etc.).
- Non-diagnostic explanation: output reflects visual resemblance to dataset labels only.

### Section 3: Symptom Recording & Voice Typing
- **Bilingual Voice Typing**: Speak symptoms directly into the browser in either **English (`en-US`)** or **Hindi (`hi-IN`)** via the Web Speech API.
- **Symptom Separation**: Automatically or manually parses spoken speech and entered notes into discrete, structured observation cards.
- **Quick-Add Common Symptoms**: Instant chips for common symptoms (`Dry Cough`, `Fever`, `Shortness of Breath ⚠️`, `Chest Pain ⚠️`, `Fatigue`, `Night Sweats`, `Coughing Blood 🚨`).
- **Physiological Domain Categorization**: Groups observations into medical physiological domains (`Respiratory / श्वसन`, `Constitutional & Systemic / दैहिक व सामान्य`, `Cardiovascular & Thoracic / हृदय व वक्षीय`, `Gastrointestinal & Other`).
- **Red-Flag Clinical Warning**: Automatically identifies urgent clinical symptoms (dyspnea, hemoptysis, acute chest pain, persistent high fever) and flags them with high-visibility alerts.
- **Physician Consultation Discussion Points**: Generates tailored questions to help patients organize their clinical visit.
- **Non-Diagnostic Educational Boundaries**: Explicitly disclaims automated disease diagnosis or treatment prescribing.

### Section 4: Combined Findings Summary & Bilingual Voice Assistant
- Consolidates session observations across distinct, clearly labeled cards:
  - **Card 1**: `🧪 Report-Extracted Facts (Source: Uploaded Lab Report)`
  - **Card 2**: `🫁 Model Output (Source: Pretrained ResNet18 Baseline)`
  - **Card 3**: `📝 Recorded Symptoms & Domains (Source: User Voice & Input)`
  - **Card 4**: `🛡️ Clinical Boundaries & Non-Merging Notice`
- **Interactive AI Voice Assistant**:
  - Allows anyone to listen to a fluent, personalized summary walkthrough of their report in voice.
  - **Bilingual Audio Support**: Native speech synthesis in **English** and **Hindi (`हिंदी`)**.
  - **Controls**: Read Report Aloud (`▶️`), Pause (`⏸️`), Stop (`⏹️`), and Speed adjustment (`0.85x`, `1.0x`, `1.2x`).
  - **Animated Visualizer**: 8-bar dynamic equalizer waveform that pulses during active playback.
  - **Spoken Narrative Script Viewer**: Displays the exact transcript read aloud by the assistant in English or Devanagari Hindi.
- **Non-Merging Guarantee**: The sources are maintained as independent panels and are **NEVER merged into a composite numerical risk score**.
- Interactive empty states with quick-action buttons (`+ Upload Lab Report`, `+ Upload Chest X-Ray`, `+ Record Symptoms with Voice`).
- Complete citations for dataset sources and active prototype scope.
- Live local dataset status and privacy verification checklist.

---

## 3. Step-by-Step Demo Checklist

Follow these steps to conduct the local demonstration:

- [ ] **Step 1: Start the application**
  - Run `.\.venv\Scripts\python.exe app.py`.
  - Open `http://127.0.0.1:5000` in Google Chrome or Microsoft Edge.
  - Confirm the API status dot in the sidebar footer displays `API online · Model ready`.
  - Confirm the top persistent safety ribbon is visible.

- [ ] **Step 2: Section 1 — Lab Report Analysis & Review**
  - Ensure Section 1 (`1. Lab Report`) is active.
  - Review the **Privacy-First & Local Only** notice.
  - Review the **Plain-Language Definitions Card** (Test Value, Unit of Measurement, Reference Range) and the **Critical Clinical Reminder**.
  - Drag and drop or choose `data/raw/Blood Test Reference Ranges.pdf`.
  - Observe the file preview (filename, size).
  - Click **🔍 Analyse Report**. Observe the upload progress bar (`Uploading to local RAM…` -> `Parsing PDF in memory…` -> `Extracted ✓`).
  - Review the structured table containing extracted test names, values, units, reference intervals, flags, and plain explanations.

- [ ] **Step 3: Section 2 — Chest X-Ray Baseline Model**
  - Click `2. Chest X-Ray` in the sidebar or click `Proceed to Chest X-Ray Model →`.
  - Review the medical disclaimer stating output is an experimental screening score, not a medical diagnosis.
  - Inspect the **ResNet18 Model Evaluation Card** showing held-out test split metrics (99.5% accuracy, 98.1% sensitivity, 99.8% specificity, 99.0% precision, 98.6% F1, 0.9991 ROC-AUC) and the confusion matrix table.
  - Test sample image **Normal-1**: click `Normal-1` button under the sample picker. Observe score `< 0.50`, predicted class `Normal`, and green probability bar.
  - Test sample image **Tuberculosis-1**: click `Tuberculosis-1` button. Observe score `1.0000` (`>= 0.50`), predicted class `TB-consistent findings`, and elevated screening score badge.
  - Test upload: choose any PNG/JPG X-ray image and click **🔍 Analyse X-Ray**.

- [ ] **Step 4: Section 3 — Voice Typing & Symptom Recording**
  - Click `3. Symptom Notes` in the sidebar.
  - Test **Language Selection**: Toggle between `🇬🇧 English` and `🇮🇳 हिंदी` to adapt the voice typing interface.
  - Click **🗣️ Try Sample Voice Input** (or click the microphone button to dictate).
  - Observe that symptoms are automatically separated into discrete symptom cards.
  - Click quick-add pills such as `+ Shortness of Breath ⚠️` and `+ Fever`.
  - Click **🔬 Analyse Recorded Symptoms**:
    - Observe the **Summary Metrics** (Symptoms count, Domains, Red Flags).
    - Review the **Red-Flag Warning Banner** highlighting critical symptoms.
    - Inspect the **Physiological Domains Breakdown** (Respiratory, Constitutional, etc.).
    - Review the **Suggested Physician Consultation Talking Points**.

- [ ] **Step 5: Section 4 — Combined Findings Summary & Voice Assistant**
  - Click `4. Findings Summary` in the sidebar.
  - At the top of Section 4, view the **Voice Assistant Card**.
  - Click **▶️ Read Report Aloud**:
    - Notice the status updates to `Speaking in English…`.
    - Watch the animated 8-bar audio waveform visualizer dance during playback.
    - Test **Pause (`⏸️`)**, **Resume (`▶️`)**, and **Speed Pills (`1.0x`, `1.2x`)**.
  - Switch the language toggle to **`🇮🇳 हिंदी वॉयस (Hindi)`**:
    - Notice the transcript accordion immediately displays the Devanagari Hindi summary narrative.
    - Click **▶️ रिपोर्ट सुनें** to hear the report read aloud in natural Hindi.
  - Scroll down to review all four findings cards:
    - `🧪 Report-Extracted Facts (Source: Uploaded Lab Report)`
    - `🫁 Model Output (Source: Pretrained ResNet18 Baseline)`
    - `📝 Recorded Symptoms & Domains (Source: User Voice & Input)`
    - `🛡️ Clinical Boundaries & Non-Merging Notice`
  - Review the **Local Dataset Status & Privacy Verification** checklist.

---

## 4. Held-Out Test Set Performance Metrics

The baseline model was trained on a stratified 80% train / 5% val split of the verified 4,200 images in the *TB Chest Radiography Database* (`seed = 42`) and evaluated **strictly on the untouched 15% held-out test split** (630 images: 525 Normal, 105 Tuberculosis).

| Metric | Held-Out Test Score | Evaluation Formula / Definition |
|---|---|---|
| **Accuracy** | **99.52%** (0.9952) | $(TP + TN) / \text{Total} = 627 / 630$ |
| **Sensitivity (Recall)** | **98.10%** (0.9810) | $TP / (TP + FN) = 103 / 105$ (TB-consistent detection rate) |
| **Specificity** | **99.81%** (0.9981) | $TN / (TN + FP) = 524 / 525$ (Normal exclusion rate) |
| **Precision** | **99.04%** (0.9904) | $TP / (TP + FP) = 103 / 104$ |
| **F1-Score** | **98.56%** (0.9856) | $2 \times (\text{Precision} \times \text{Recall}) / (\text{Precision} + \text{Recall})$ |
| **ROC-AUC** | **0.9991** | Area under the Receiver Operating Characteristic curve |

### Actual Confusion Matrix (630 Held-Out Test Images)

```
                       Predicted Normal    Predicted TB-Consistent    Total
Actual Normal (525)    TN = 524            FP = 1                     525
Actual TB (105)        FN = 2              TP = 103                   105
Total                  526                 104                        630
```

- **True Negatives (TN = 524)**: Normal radiographs correctly categorized as Normal.
- **False Positives (FP = 1)**: Normal radiograph flagged with TB-consistent findings.
- **False Negatives (FN = 2)**: Radiographs with TB-consistent findings predicted as Normal.
- **True Positives (TP = 103)**: Radiographs with TB-consistent findings correctly identified.

### Patient Grouping Scope Limitation
Source metadata files (`Normal.metadata.xlsx`, `Tuberculosis.metadata.xlsx`) do not provide patient IDs, hospital encounter numbers, age, or sex. Evaluation is strictly image-level across 4,200 radiographs. **Independent patient-level evaluation cannot be established and is not claimed.**

---

## 5. Explicitly Deferred Features (One-Day Prototype Scope)

To maintain safety, rigorous verification, and zero hallucination within the single-day prototype scope, the following capabilities are **explicitly deferred**:

1. **Voice / native-language input**: Deferred to prevent transcription inaccuracies and non-deterministic translation errors.
2. **Symptom NLP**: Deferred; notes are strictly user-entered and unanalyzed to prevent speculative AI diagnostic leaps.
3. **OCR on arbitrary report layouts beyond prototype**: Deferred to avoid misparsing unstructured heterogeneous commercial lab layouts.
4. **Knowledge graph**: Deferred; no external ontological disease-relation mapping.
5. **Cross-modal clinical risk scoring**: Deferred; outputs are never combined into a composite clinical risk score.
6. **Heatmap explanations (Grad-CAM)**: Deferred; saliency maps can produce misleading clinical reassurance.
7. **Automated specialist routing**: Deferred; patient triage must be managed by licensed clinicians.
8. **Medication safety alerts**: Deferred; prescription drug interactions require validated clinical pharmacopeia engines.
9. **Diabetic retinopathy and oral cancer screening**: Deferred; out of scope for the current pulmonary and routine blood panel prototype.

---

## 6. Citations & Dataset Attributions

### 1. NLM Montgomery & Shenzhen Chest X-Ray Datasets
> Jaeger S, Karargyris A, Candemir S, Folio L, Siegelman J, Callaghan F, Xue Z, Dang C, Chou P, Prevedello LM, Lu PX, Antani S.  
> *"Two public chest X-ray datasets for computer-aided screening of pulmonary diseases."*  
> **Quantitative Imaging in Medicine and Surgery**, 2014;4(6):475-477.  
> [doi:10.3978/j.issn.2223-4292.2014.11.20](https://doi.org/10.3978/j.issn.2223-4292.2014.11.20)  
> *Collaboration between Shenzhen No. 3 People's Hospital (Guangdong, China) and National Library of Medicine (NIH, Bethesda, MD, USA).*

### 2. TB Chest Radiography Database
> Rahman T, Khandakar A, Kadir MA, Islam KR, Islam KF, Mazhar R, Hamid T, Islam MT, Kashem S, Mahbub ZB, Ayari MA, Chowdhury MEH.  
> *"Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization."*  
> **IEEE Access**, 2020;8:191586-191601.  
> [doi:10.1109/ACCESS.2020.3031384](https://doi.org/10.1109/ACCESS.2020.3031384)

### 3. Sample Laboratory Report & Reference Ranges
> Synthetic lab report documents and reference intervals compiled from standard adult clinical pathology references (e.g. *Tietz Textbook of Clinical Chemistry and Molecular Diagnostics*). Provided strictly for non-clinical software testing and pair-programming development.

---

## 7. Installation & Startup Instructions (Any Device / OS)

The application runs entirely locally on **Windows**, **macOS**, and **Linux**.

### Prerequisites
- **Python**: Version `3.10` or higher (tested with Python 3.10, 3.11, 3.12, and 3.13)
- **Modern Web Browser**: Google Chrome, Microsoft Edge, Mozilla Firefox, or Apple Safari
- *(Optional)* **Tesseract OCR**: Required only if you wish to parse scanned/raster PDF reports or image-only lab photos via OCR:
  - **Windows**: Download the installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) or install via winget: `winget install UB-Mannheim.TesseractOCR`
  - **macOS**: `brew install tesseract`
  - **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install -y tesseract-ocr`

---

### Step-by-Step Setup

#### Step 1: Open Terminal / Shell
Navigate to the project root directory:
```bash
cd "Health care dignostic sysytem"
```

#### Step 2: Create & Activate a Python Virtual Environment

- **On Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(If PowerShell script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **On Windows (Command Prompt `cmd.exe`)**:
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```

- **On macOS / Linux (bash or zsh)**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### Step 3: Install Required Dependencies
With the virtual environment active, install all required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Start the Local Flask Application Server
Run the Flask server:
- **Windows**:
  ```powershell
  python app.py
  ```
- **macOS / Linux**:
  ```bash
  python3 app.py
  ```

Expected startup log:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
 * Application ready!
```

#### Step 5: Access the Web Interface
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```
The application will load immediately in **Light Mode** (default) with options to switch between Light and Dark modes.

---

### Running the Automated Test Suite (23 Tests)
To verify that all routes, models, OCR extractors, symptom parsers, and summary endpoints are functional on your device:

- **Windows**:
  ```powershell
  python tests/test_demo.py
  ```
- **macOS / Linux**:
  ```bash
  python3 tests/test_demo.py
  ```

Expected test result:
```
== Healthcare Diagnostic Assistant - Baseline Test Suite ==
[PASS] GET /api/health -> 200 OK
[PASS] Disclaimer text present in /api/health
[PASS] GET /api/model/status -> 200 (model available & threshold=0.50)
[PASS] POST /api/analyse (no file) -> 400
[PASS] POST /api/analyse (.pdf) -> 415
[PASS] POST /api/xray/upload (valid PNG) -> 200
[PASS] POST /api/lab/parse (empty file) -> 400
[PASS] POST /api/lab/parse (corrupted PDF header) -> 400
[PASS] POST /api/xray/upload (corrupted image bytes) -> 400
[PASS] POST /api/xray/upload (model class, score/threshold, non-diagnostic explanation verified)
[PASS] POST /api/lab/parse (in-memory parsing of reference PDF) -> 200 (rows=14)
[PASS] POST /api/lab/parse (scanned PDF via Tesseract OCR) -> 200 (ocr_used=True)
[PASS] POST /api/lab/parse (image report via Tesseract OCR) -> 200 (ocr_used=True)
[PASS] GET /api/dataset/status -> 200 (xray images=4200)
[PASS] GET / -> 200 (frontend HTML served)
[PASS] GET /api/xray/model-info -> 200 (held-out test metrics & limitations verified)
[PASS] GET /api/xray/sample-images -> 200 (curated samples available)
[PASS] GET /api/xray/image -> 200 (sample image served)
[PASS] POST /api/xray/predict (Normal-1) -> score=0.0134, label=Normal
[PASS] POST /api/xray/predict (Tuberculosis-1) -> score=1.0, label=TB-consistent findings
[PASS] POST /api/symptoms/analyze -> 200 (red_flags=['Shortness of breath (Dyspnea)', 'High fever'], domains=2)
[PASS] POST /api/summary/voice-script -> 200 (EN len=801, HI len=682)

================================================
  23 passed | 0 failed
================================================
```


---

## 8. License & Non-Clinical Use Agreement

This prototype is made available under the **MIT License** for academic, research, and technical evaluation purposes only. Commercial medical use, diagnostic decision-making, patient triage, and clinical deployment are strictly prohibited.
