/**
 * client_parser.js – Client-Side (Browser Wasm) Medical Document Parser & OCR Engine
 *
 * Enhanced with:
 * 1. Clinical Decimal vs Integer Disambiguation Engine:
 *    - Knows which tests MUST have decimals (e.g. Hemoglobin 14.5, Creatinine 0.9, RBC 4.8)
 *      and auto-restores missing decimal points when OCR eats dots (e.g. 145 -> 14.5, 08 -> 0.8).
 *    - Knows which tests MUST be whole integers (e.g. Glucose 95, Cholesterol 180, ESR 15)
 *      and removes stray flyspeck dots (e.g. 9.5 -> 95, 1.85 -> 185).
 * 2. Levenshtein Fuzzy Medical Auto-Correct:
 *    - Snaps misspelled test names (e.g. "Hemogiobin" -> "Hemoglobin", "Plateiet" -> "Platelets")
 * 3. OCR Homoglyph & Unit Normalization:
 *    - Resolves 'O' <-> '0', 'l' <-> '1', ',' <-> '.'
 *    - Cleans units ('g/dI' -> 'g/dL', 'mgl/dl' -> 'mg/dL', '10*3' -> '10³/µL')
 * 4. 100% On-Device Browser Execution (PDF.js + Tesseract.js WebAssembly).
 */

(function (window) {
  "use strict";

  // ---------------------------------------------------------------------------
  // Clinical Reference Database with Decimal & Physiological Profiles
  // ---------------------------------------------------------------------------
  const CLINICAL_REFERENCE_DB = {
    // Haematology (CBC)
    "hemoglobin":           { display: "Hemoglobin", low: 12.0, high: 17.5, unit: "g/dL", is_decimal: true, typical_dec: 1, min: 4.0, max: 25.0, note: "Male 13.5–17.5 | Female 12.0–15.5" },
    "haemoglobin":          { display: "Hemoglobin", low: 12.0, high: 17.5, unit: "g/dL", is_decimal: true, typical_dec: 1, min: 4.0, max: 25.0, note: "Male 13.5–17.5 | Female 12.0–15.5" },
    "hgb":                  { display: "Hemoglobin (HGB)", low: 12.0, high: 17.5, unit: "g/dL", is_decimal: true, typical_dec: 1, min: 4.0, max: 25.0, note: "Male 13.5–17.5 | Female 12.0–15.5" },
    "rbc":                  { display: "RBC Count", low: 3.8,  high: 5.8,  unit: "10⁶/µL", is_decimal: true, typical_dec: 2, min: 1.5, max: 9.0, note: "Male 4.5–5.9 | Female 3.8–5.2" },
    "rbc count":            { display: "RBC Count", low: 3.8,  high: 5.8,  unit: "10⁶/µL", is_decimal: true, typical_dec: 2, min: 1.5, max: 9.0, note: "Male 4.5–5.9 | Female 3.8–5.2" },
    "rbc_count":            { display: "RBC Count", low: 3.8,  high: 5.8,  unit: "10⁶/µL", is_decimal: true, typical_dec: 2, min: 1.5, max: 9.0, note: "Male 4.5–5.9 | Female 3.8–5.2" },
    "wbc":                  { display: "WBC Count", low: 4.0,  high: 11.0, unit: "10³/µL", is_decimal: false, min: 500, max: 60000, note: "Normal 4,000–11,000 /µL" },
    "wbc count":            { display: "WBC Count", low: 4.0,  high: 11.0, unit: "10³/µL", is_decimal: false, min: 500, max: 60000, note: "Normal 4,000–11,000 /µL" },
    "wbc_count":            { display: "WBC Count", low: 4.0,  high: 11.0, unit: "10³/µL", is_decimal: false, min: 500, max: 60000, note: "Normal 4,000–11,000 /µL" },
    "total leucocyte count":{ display: "Total Leucocyte Count (TLC)", low: 4.0, high: 11.0, unit: "10³/µL", is_decimal: false, min: 500, max: 60000, note: "Normal 4,000–11,000 /µL" },
    "tlc":                  { display: "Total Leucocyte Count (TLC)", low: 4.0, high: 11.0, unit: "10³/µL", is_decimal: false, min: 500, max: 60000, note: "Normal 4,000–11,000 /µL" },
    "platelet count":       { display: "Platelet Count", low: 150.0, high: 400.0, unit: "10³/µL", is_decimal: false, min: 10, max: 999999, note: "Normal 1.5–4.0 Lakhs/cumm" },
    "platelet_count":       { display: "Platelet Count", low: 150.0, high: 400.0, unit: "10³/µL", is_decimal: false, min: 10, max: 999999, note: "Normal 1.5–4.0 Lakhs/cumm" },
    "platelets":            { display: "Platelet Count", low: 150.0, high: 400.0, unit: "10³/µL", is_decimal: false, min: 10, max: 999999, note: "Normal 150,000–400,000 /µL" },
    "hematocrit":           { display: "Hematocrit (PCV)", low: 36.0, high: 52.0, unit: "%", is_decimal: true, typical_dec: 1, min: 15.0, max: 70.0, note: "Male 41–53 | Female 36–46" },
    "haematocrit":          { display: "Hematocrit (PCV)", low: 36.0, high: 52.0, unit: "%", is_decimal: true, typical_dec: 1, min: 15.0, max: 70.0, note: "Male 41–53 | Female 36–46" },
    "pcv":                  { display: "Packed Cell Volume (PCV)", low: 36.0, high: 52.0, unit: "%", is_decimal: true, typical_dec: 1, min: 15.0, max: 70.0, note: "Packed Cell Volume" },
    "mcv":                  { display: "MCV", low: 80.0, high: 100.0, unit: "fL", is_decimal: true, typical_dec: 1, min: 50.0, max: 130.0, note: "Mean Corpuscular Volume" },
    "mch":                  { display: "MCH", low: 27.0, high: 33.0, unit: "pg", is_decimal: true, typical_dec: 1, min: 15.0, max: 45.0, note: "Mean Corpuscular Hemoglobin" },
    "mchc":                 { display: "MCHC", low: 32.0, high: 36.0, unit: "g/dL", is_decimal: true, typical_dec: 1, min: 20.0, max: 45.0, note: "Mean Corpuscular Hb Concentration" },
    "esr":                  { display: "ESR", low: 0.0, high: 20.0, unit: "mm/hr", is_decimal: false, min: 0, max: 150, note: "Male 0–15 | Female 0–20 mm/hr" },

    // Biochemistry & Blood Sugar (Integers, except HbA1c)
    "glucose":              { display: "Blood Glucose (Fasting)", low: 70.0, high: 99.0, unit: "mg/dL", is_decimal: false, min: 30, max: 600, note: "Fasting normal 70–99 mg/dL" },
    "fasting blood sugar":  { display: "Fasting Blood Sugar (FBS)", low: 70.0, high: 99.0, unit: "mg/dL", is_decimal: false, min: 30, max: 600, note: "Fasting normal 70–99 mg/dL" },
    "fasting_blood_sugar":  { display: "Fasting Blood Sugar (FBS)", low: 70.0, high: 99.0, unit: "mg/dL", is_decimal: false, min: 30, max: 600, note: "Fasting normal 70–99 mg/dL" },
    "fbs":                  { display: "Fasting Blood Sugar (FBS)", low: 70.0, high: 99.0, unit: "mg/dL", is_decimal: false, min: 30, max: 600, note: "Fasting blood sugar" },
    "post prandial blood sugar": { display: "Post Prandial Blood Sugar (PPBS)", low: 70.0, high: 140.0, unit: "mg/dL", is_decimal: false, min: 40, max: 700, note: "PPBS normal < 140 mg/dL" },
    "ppbs":                 { display: "Post Prandial Blood Sugar (PPBS)", low: 70.0, high: 140.0, unit: "mg/dL", is_decimal: false, min: 40, max: 700, note: "PPBS normal < 140 mg/dL" },
    "hba1c":                { display: "HbA1c (Glycated Hemoglobin)", low: 4.0, high: 5.6, unit: "%", is_decimal: true, typical_dec: 1, min: 3.5, max: 18.0, note: "Non-diabetic: < 5.7%" },

    // Lipid Profile (All Integers)
    "total cholesterol":    { display: "Total Cholesterol", low: 0.0, high: 200.0, unit: "mg/dL", is_decimal: false, min: 50, max: 600, note: "Desirable < 200 mg/dL" },
    "total_cholesterol":    { display: "Total Cholesterol", low: 0.0, high: 200.0, unit: "mg/dL", is_decimal: false, min: 50, max: 600, note: "Desirable < 200 mg/dL" },
    "cholesterol":          { display: "Total Cholesterol", low: 0.0, high: 200.0, unit: "mg/dL", is_decimal: false, min: 50, max: 600, note: "Desirable < 200 mg/dL" },
    "ldl":                  { display: "LDL Cholesterol", low: 0.0, high: 100.0, unit: "mg/dL", is_decimal: false, min: 20, max: 400, note: "Optimal < 100 mg/dL" },
    "ldl cholesterol":      { display: "LDL Cholesterol", low: 0.0, high: 100.0, unit: "mg/dL", is_decimal: false, min: 20, max: 400, note: "Optimal < 100 mg/dL" },
    "hdl":                  { display: "HDL Cholesterol", low: 40.0, high: 9999.0, unit: "mg/dL", is_decimal: false, min: 10, max: 150, note: "Male ≥ 40 | Female ≥ 50" },
    "hdl cholesterol":      { display: "HDL Cholesterol", low: 40.0, high: 9999.0, unit: "mg/dL", is_decimal: false, min: 10, max: 150, note: "Male ≥ 40 | Female ≥ 50" },
    "triglycerides":        { display: "Triglycerides", low: 0.0, high: 150.0, unit: "mg/dL", is_decimal: false, min: 30, max: 1000, note: "Normal < 150 mg/dL" },
    "vldl":                 { display: "VLDL", low: 2.0, high: 30.0, unit: "mg/dL", is_decimal: false, min: 2, max: 120, note: "Normal 2–30 mg/dL" },

    // Renal / KFT
    "creatinine":           { display: "Serum Creatinine", low: 0.6, high: 1.2, unit: "mg/dL", is_decimal: true, typical_dec: 1, min: 0.2, max: 15.0, note: "Male 0.7–1.3 | Female 0.6–1.1 mg/dL" },
    "serum creatinine":     { display: "Serum Creatinine", low: 0.6, high: 1.2, unit: "mg/dL", is_decimal: true, typical_dec: 1, min: 0.2, max: 15.0, note: "Male 0.7–1.3 | Female 0.6–1.1 mg/dL" },
    "blood urea":           { display: "Blood Urea", low: 15.0, high: 45.0, unit: "mg/dL", is_decimal: false, min: 5, max: 200, note: "Normal 15–45 mg/dL" },
    "urea":                 { display: "Blood Urea", low: 15.0, high: 45.0, unit: "mg/dL", is_decimal: false, min: 5, max: 200, note: "Normal 15–45 mg/dL" },
    "bun":                  { display: "BUN (Blood Urea Nitrogen)", low: 7.0, high: 20.0, unit: "mg/dL", is_decimal: false, min: 3, max: 120, note: "Blood Urea Nitrogen 7–20 mg/dL" },
    "uric acid":            { display: "Uric Acid", low: 3.5, high: 7.2, unit: "mg/dL", is_decimal: true, typical_dec: 1, min: 1.0, max: 20.0, note: "Male 4.0–8.0 | Female 2.5–7.0 mg/dL" },

    // Liver / LFT
    "ast":                  { display: "SGOT (AST)", low: 10.0, high: 40.0, unit: "U/L", is_decimal: false, min: 5, max: 1000, note: "SGOT (AST)" },
    "sgot":                 { display: "SGOT (AST)", low: 10.0, high: 40.0, unit: "U/L", is_decimal: false, min: 5, max: 1000, note: "SGOT (AST)" },
    "alt":                  { display: "SGPT (ALT)", low: 7.0, high: 56.0, unit: "U/L", is_decimal: false, min: 5, max: 1000, note: "SGPT (ALT)" },
    "sgpt":                 { display: "SGPT (ALT)", low: 7.0, high: 56.0, unit: "U/L", is_decimal: false, min: 5, max: 1000, note: "SGPT (ALT)" },
    "alkaline phosphatase": { display: "Alkaline Phosphatase (ALP)", low: 44.0, high: 147.0, unit: "U/L", is_decimal: false, min: 20, max: 800, note: "Normal adult 44–147 U/L" },
    "alp":                  { display: "Alkaline Phosphatase (ALP)", low: 44.0, high: 147.0, unit: "U/L", is_decimal: false, min: 20, max: 800, note: "Normal adult 44–147 U/L" },
    "bilirubin total":      { display: "Total Bilirubin", low: 0.2, high: 1.2, unit: "mg/dL", is_decimal: true, typical_dec: 1, min: 0.1, max: 25.0, note: "Total Bilirubin 0.2–1.2 mg/dL" },
    "total bilirubin":      { display: "Total Bilirubin", low: 0.2, high: 1.2, unit: "mg/dL", is_decimal: true, typical_dec: 1, min: 0.1, max: 25.0, note: "Total Bilirubin 0.2–1.2 mg/dL" },
    "direct bilirubin":     { display: "Direct Bilirubin", low: 0.0, high: 0.3, unit: "mg/dL", is_decimal: true, typical_dec: 1, min: 0.0, max: 15.0, note: "Direct Bilirubin 0.0–0.3 mg/dL" },
    "total protein":        { display: "Total Protein", low: 6.0, high: 8.3, unit: "g/dL", is_decimal: true, typical_dec: 1, min: 3.0, max: 12.0, note: "Normal 6.0–8.3 g/dL" },
    "albumin":              { display: "Albumin", low: 3.5, high: 5.0, unit: "g/dL", is_decimal: true, typical_dec: 1, min: 1.5, max: 7.0, note: "Normal 3.5–5.0 g/dL" },

    // Electrolytes & Thyroid
    "calcium":              { display: "Serum Calcium", low: 8.5, high: 10.5, unit: "mg/dL", is_decimal: true, typical_dec: 1, min: 4.0, max: 16.0, note: "Serum Calcium 8.5–10.5 mg/dL" },
    "potassium":            { display: "Serum Potassium", low: 3.5, high: 5.1, unit: "mEq/L", is_decimal: true, typical_dec: 1, min: 1.5, max: 9.0, note: "Serum Potassium 3.5–5.1 mEq/L" },
    "sodium":               { display: "Serum Sodium", low: 135.0, high: 145.0, unit: "mEq/L", is_decimal: false, min: 100, max: 180, note: "Serum Sodium 135–145 mEq/L" },
    "tsh":                  { display: "TSH (Thyroid Stimulating Hormone)", low: 0.4, high: 4.5, unit: "µIU/mL", is_decimal: true, typical_dec: 2, min: 0.01, max: 50.0, note: "Thyroid Stimulating Hormone" },
  };

  const UNCERTAIN_NOTE = "Please check the original report.";
  const CLINICAL_DISCLAIMER =
    "⚠️ Reference intervals vary across laboratories, clinical instruments, patient age, and biological sex. " +
    "Printed values on your report always supersede general reference estimates. " +
    "This system provides educational decision support only and cannot diagnose or rule out any health condition. " +
    "Discuss all laboratory findings with a qualified healthcare physician.";

  // ---------------------------------------------------------------------------
  // Levenshtein Fuzzy String Matching (Auto-Corrects OCR Typos)
  // ---------------------------------------------------------------------------
  function levenshteinDistance(s1, s2) {
    const m = s1.length, n = s2.length;
    const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
        dp[i][j] = Math.min(
          dp[i - 1][j] + 1,
          dp[i][j - 1] + 1,
          dp[i - 1][j - 1] + cost
        );
      }
    }
    return dp[m][n];
  }

  function getSimilarity(s1, s2) {
    const maxLen = Math.max(s1.length, s2.length);
    if (maxLen === 0) return 1.0;
    return 1 - (levenshteinDistance(s1, s2) / maxLen);
  }

  function fuzzyMatchTestName(rawName) {
    if (!rawName) return null;
    const clean = rawName.toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
    if (!clean) return null;

    // 1. Direct or substring match
    for (const [key, ref] of Object.entries(CLINICAL_REFERENCE_DB)) {
      if (clean === key || clean.startsWith(key) || key.startsWith(clean)) {
        return { key, display: ref.display, sim: 1.0 };
      }
    }

    // 2. Strip common clinical prefixes: "serum", "total", "blood", "fasting", "s.", "random"
    const stripped = clean.replace(/^(serum|total|blood|fasting|random|post prandial|s\s)\s+/g, "").trim();

    // 3. Fuzzy Levenshtein match
    let bestKey = null;
    let bestSim = 0;
    for (const [key, ref] of Object.entries(CLINICAL_REFERENCE_DB)) {
      const s1 = getSimilarity(clean, key);
      const s2 = stripped ? getSimilarity(stripped, key) : 0;
      const sim = Math.max(s1, s2);
      if (sim > bestSim && sim >= 0.74) {
        bestSim = sim;
        bestKey = key;
      }
    }

    if (bestKey) {
      return { key: bestKey, display: CLINICAL_REFERENCE_DB[bestKey].display, sim: bestSim };
    }
    return null;
  }

  // ---------------------------------------------------------------------------
  // Decimal vs Integer Auto-Correction & Homoglyph Normalization
  // ---------------------------------------------------------------------------
  function repairValueDecimal(testKey, rawVal) {
    if (!rawVal) return rawVal;

    // 1. Homoglyph cleaning in numeric string
    let val = String(rawVal).trim();
    // Replace letter o/O with 0
    val = val.replace(/[oO]/g, "0");
    // Replace l, I, | with 1
    val = val.replace(/[lI|]/g, "1");
    // Replace comma with dot decimal separator
    val = val.replace(/,/g, ".");
    // Strip trailing/leading non-digits except dot
    val = val.replace(/^[^\d.]+/, "").replace(/[^\d.]+$/, "");

    const ref = testKey ? CLINICAL_REFERENCE_DB[testKey] : null;
    if (!ref) {
      return val;
    }

    const numVal = parseFloat(val);

    // Case A: Test is EXPECTED TO BE A DECIMAL (e.g. Hemoglobin 14.5, Creatinine 0.9, RBC 4.8)
    if (ref.is_decimal) {
      // If it already has a decimal point and falls within reasonable bounds
      if (val.includes(".")) {
        const parts = val.split(".");
        if (parts.length === 2 && !isNaN(numVal) && numVal >= ref.min && numVal <= ref.max) {
          return val;
        }
        // If multiple dots like "1.4.5" -> "14.5"
        if (parts.length > 2) {
          const joined = parts.slice(0, -1).join("") + "." + parts[parts.length - 1];
          if (!isNaN(parseFloat(joined))) return joined;
        }
      }

      // Missing decimal point restoration!
      const digitsOnly = val.replace(/\D/g, "");
      if (digitsOnly.length >= 2) {
        // Try multiple decimal position splits: typical_dec, 1, or 2
        for (const dec of [ref.typical_dec || 1, 1, 2]) {
          if (digitsOnly.length > dec) {
            const candidate1 = digitsOnly.slice(0, -dec) + "." + digitsOnly.slice(-dec);
            const c1Num = parseFloat(candidate1);
            if (!isNaN(c1Num) && c1Num >= ref.min && c1Num <= ref.max) {
              return candidate1;
            }
          }
        }

        // Handle leading zero: "08" -> "0.8" or "09" -> "0.9"
        if (digitsOnly.startsWith("0") && digitsOnly.length === 2) {
          const candidate2 = "0." + digitsOnly[1];
          const c2Num = parseFloat(candidate2);
          if (!isNaN(c2Num) && c2Num >= ref.min && c2Num <= ref.max) {
            return candidate2;
          }
        }
      }
    }

    // Case B: Test is EXPECTED TO BE AN INTEGER (e.g. Glucose 95, Cholesterol 180, ESR 15)
    if (ref.is_decimal === false) {
      // Remove stray flyspeck dots (e.g. "9.5" for Glucose 95, or "1.85" for Cholesterol 185)
      if (val.includes(".")) {
        const withoutDot = val.replace(/\./g, "");
        const wNum = parseFloat(withoutDot);
        if (!isNaN(numVal) && numVal < ref.min && !isNaN(wNum) && wNum >= ref.min && wNum <= ref.max) {
          return withoutDot;
        }
        if (val.endsWith(".0") || val.endsWith(".00")) {
          return val.split(".")[0];
        }
      }
    }

    return val;
  }

  // ---------------------------------------------------------------------------
  // Unit & Flag Normalization
  // ---------------------------------------------------------------------------
  function normalizeUnit(unitStr, testKey) {
    if (!unitStr || unitStr === UNCERTAIN_NOTE) {
      if (testKey && CLINICAL_REFERENCE_DB[testKey]) {
        return CLINICAL_REFERENCE_DB[testKey].unit;
      }
      return UNCERTAIN_NOTE;
    }
    let u = unitStr.trim();
    // Normalize g/dL
    if (/^g\/d[li1]$/i.test(u)) return "g/dL";
    // Normalize mg/dL
    if (/^mg[l\/]?\/d[li1]$/i.test(u) || /^mg\/dl$/i.test(u)) return "mg/dL";
    // Normalize 10^3/uL
    if (/^(10[\*\^]3|103)[\/]?u?l?$/i.test(u)) return "10³/µL";
    // Normalize 10^6/uL
    if (/^(10[\*\^]6|106)[\/]?u?l?$/i.test(u)) return "10⁶/µL";
    // Normalize fL
    if (/^f[li1]$/i.test(u)) return "fL";
    // Normalize mm/hr
    if (/^mm\/(hr|h|1hr)$/i.test(u)) return "mm/hr";
    // Normalize U/L
    if (/^u\/[li1]$/i.test(u)) return "U/L";
    return u;
  }

  function normalizeFlag(flag) {
    if (!flag) return "";
    const f = flag.trim().toUpperCase();
    if (f === "H" || f === "HIGH" || f === "ABOVE" || f === "HLGH") return "HIGH";
    if (f === "L" || f === "LOW" || f === "BELOW" || f === "L0W") return "LOW";
    if (f === "NORMAL" || f === "N" || f === "N0RMAL") return "NORMAL";
    if (f === "*" || f === "ABNORMAL" || f === "CRITICAL") return "ABNORMAL";
    return f;
  }

  // ---------------------------------------------------------------------------
  // Canvas Image Preprocessing (Improves OCR Contrast & DPI)
  // ---------------------------------------------------------------------------
  function preprocessImageCanvas(canvas) {
    const ctx = canvas.getContext("2d");
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const d = imgData.data;

    let min = 255;
    let max = 0;
    for (let i = 0; i < d.length; i += 4) {
      const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      if (lum < min) min = lum;
      if (lum > max) max = lum;
    }

    const range = max - min || 1;
    for (let i = 0; i < d.length; i += 4) {
      const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      // Contrast stretch
      const stretched = ((lum - min) / range) * 255;
      // High-contrast binarization
      const val = stretched < 145 ? Math.max(0, stretched - 30) : 255;
      d[i] = val;
      d[i + 1] = val;
      d[i + 2] = val;
    }
    ctx.putImageData(imgData, 0, 0);
    return canvas;
  }

  // ---------------------------------------------------------------------------
  // Parse Lines Spatial Reconstruction from PDF.js TextContent
  // ---------------------------------------------------------------------------
  function reconstructLinesFromPdfContent(textContent) {
    const items = (textContent.items || [])
      .map(it => ({
        str: it.str || "",
        x: it.transform ? it.transform[4] : 0,
        y: it.transform ? it.transform[5] : 0,
        width: it.width || 0,
        height: it.height || 0,
      }))
      .filter(it => it.str.trim().length > 0);

    if (!items.length) return [];

    items.sort((a, b) => b.y - a.y || a.x - b.x);

    const lines = [];
    let curLine = [items[0]];
    let curY = items[0].y;

    for (let i = 1; i < items.length; i++) {
      const it = items[i];
      if (Math.abs(it.y - curY) <= 5.5) {
        curLine.push(it);
      } else {
        curLine.sort((a, b) => a.x - b.x);
        lines.push(curLine);
        curLine = [it];
        curY = it.y;
      }
    }
    if (curLine.length) {
      curLine.sort((a, b) => a.x - b.x);
      lines.push(curLine);
    }

    return lines.map(lineItems => {
      let str = "";
      let lastX = null;
      let lastW = 0;
      for (const it of lineItems) {
        if (lastX !== null) {
          const gap = it.x - (lastX + lastW);
          if (gap > 18) {
            str += "   ";
          } else if (gap > 2) {
            str += " ";
          }
        }
        str += it.str;
        lastX = it.x;
        lastW = it.width;
      }
      return str.trim();
    });
  }

  // ---------------------------------------------------------------------------
  // Regex Patterns
  // ---------------------------------------------------------------------------
  const ROW_RE = new RegExp(
    "^([A-Za-z][\\w\\s/(),.\\-%·]{1,60}?)" +
    "\\s{2,}" +
    "([0-9]+(?:[.,][0-9]+)?)" +
    "\\s+" +
    "([A-Za-z0-9/%µ·^\\-]{1,20}|/cumm|lakhs/cumm|cells/\\w+)" +
    "\\s{2,}" +
    "\\(?(?:ref(?:erence)?\\.?\\s*(?:range|interval)?|normal)?\\.?:?\\s*" +
    "([0-9]+(?:\\.[0-9]+)?\\s*[\\-–—to]+\\s*[0-9]+(?:\\.[0-9]+)?|[<>≤≥]\\s*[0-9]+(?:\\.[0-9]+)?)" +
    "\\)?" +
    "(?:\\s+(H|L|High|Low|Normal|Abnormal|Above|Below|Critical|\\*))?",
    "i"
  );

  const LINE_RE = new RegExp(
    "^([A-Za-z][\\w\\s/(),.\\-%·]{2,50}?)" +
    "[\\s:=]+" +
    "([0-9]+(?:[.,][0-9]+)?)" +
    "\\s*" +
    "([A-Za-z0-9/%µ·^\\-]{1,20}|/cumm|lakhs/cumm)?" +
    "(?:\\s*\\(?(?:ref(?:erence)?\\.?\\s*(?:range|interval)?|normal)?\\.?:?\\s*" +
    "([0-9]+(?:\\.[0-9]+)?\\s*[\\-–—to]+\\s*[0-9]+(?:\\.[0-9]+)?|[<>≤≥]\\s*[0-9]+(?:\\.[0-9]+)?)" +
    "\\)?)?" +
    "(?:\\s*(H|L|High|Low|Normal|Abnormal|\\*))?",
    "i"
  );

  function cleanString(s) {
    if (!s) return UNCERTAIN_NOTE;
    const clean = s.replace(/\s{2,}/g, " ").trim();
    return clean || UNCERTAIN_NOTE;
  }

  function parseNumericRange(refStr) {
    if (!refStr) return null;
    const dashMatch = refStr.match(/([0-9]+(?:\.[0-9]+)?)\s*[\-–—to]+\s*([0-9]+(?:\.[0-9]+)?)/i);
    if (dashMatch) {
      return { low: parseFloat(dashMatch[1]), high: parseFloat(dashMatch[2]) };
    }
    const maxMatch = refStr.match(/[<≤]\s*([0-9]+(?:\.[0-9]+)?)/);
    if (maxMatch) {
      return { low: 0, high: parseFloat(maxMatch[1]) };
    }
    const minMatch = refStr.match(/[>≥]\s*([0-9]+(?:\.[0-9]+)?)/);
    if (minMatch) {
      return { low: parseFloat(minMatch[1]), high: 99999 };
    }
    return null;
  }

  function calculateFlag(valNum, refRangeStr, explicitFlag, testKey) {
    if (explicitFlag) {
      const norm = normalizeFlag(explicitFlag);
      if (norm) return norm;
    }

    const parsedRange = parseNumericRange(refRangeStr);
    if (parsedRange && !isNaN(valNum)) {
      if (valNum < parsedRange.low) return "LOW";
      if (parsedRange.high < 90000 && valNum > parsedRange.high) return "HIGH";
      return "NORMAL";
    }

    if (testKey && CLINICAL_REFERENCE_DB[testKey] && !isNaN(valNum)) {
      const ref = CLINICAL_REFERENCE_DB[testKey];
      if (valNum < ref.low) return "LOW";
      if (ref.high < 9000 && valNum > ref.high) return "HIGH";
      return "NORMAL";
    }

    return "";
  }

  function generatePlainExplanation(row) {
    const flag = (row.flag || "").toUpperCase();
    const ref = row.ref_range;
    const isRefMissing = !ref || ref === UNCERTAIN_NOTE || ref === "—" || ref === "Not specified";

    if (isRefMissing) {
      if (flag === "HIGH") return "High: Elevated level";
      if (flag === "LOW") return "Low: Below standard level";
      return "Normal";
    }

    if (flag === "HIGH") {
      return "High: Above expected range";
    }
    if (flag === "LOW") {
      return "Low: Below expected range";
    }
    if (flag === "NORMAL") {
      return "Normal: Within expected range";
    }
    return "Normal";
  }

  // ---------------------------------------------------------------------------
  // Primary Parser Function with Fuzzy Name & Decimal Auto-Correction
  // ---------------------------------------------------------------------------
  function parseTextIntoStructuredRows(text) {
    const rawLines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length >= 4);
    const rows = [];
    const seen = new Set();

    for (const line of rawLines) {
      // 1. Strict multi-column pattern
      let m = line.match(ROW_RE);
      if (m) {
        const [, name, val, unit, range, flag] = m;
        const fuzzy = fuzzyMatchTestName(name);
        const testKey = fuzzy ? fuzzy.key : cleanString(name).toLowerCase();
        const testDisplayName = fuzzy ? fuzzy.display : cleanString(name);

        // Auto-correct decimal errors based on clinical domain rules
        const correctedVal = repairValueDecimal(testKey, val);
        const valNum = parseFloat(correctedVal);
        const correctedUnit = normalizeUnit(unit, testKey);
        const cleanRange = cleanString(range);
        const computedFlag = calculateFlag(valNum, cleanRange, flag, testKey);

        const row = {
          test_name: testDisplayName,
          value: correctedVal,
          unit: correctedUnit,
          ref_range: cleanRange,
          flag: computedFlag,
          source_line: line,
        };
        row.plain_explanation = generatePlainExplanation(row);

        const dedupKey = `${testKey}_${row.value}`;
        if (!seen.has(dedupKey)) {
          seen.add(dedupKey);
          rows.push(row);
        }
        continue;
      }

      // 2. Looser colon/delimited pattern
      m = line.match(LINE_RE);
      if (m) {
        const [, name, val, unit, range, flag] = m;
        if ((name || "").trim().length < 3) continue;

        const fuzzy = fuzzyMatchTestName(name);
        const testKey = fuzzy ? fuzzy.key : cleanString(name).toLowerCase();
        const testDisplayName = fuzzy ? fuzzy.display : cleanString(name);

        const correctedVal = repairValueDecimal(testKey, val);
        const valNum = parseFloat(correctedVal);
        const correctedUnit = normalizeUnit(unit, testKey);
        const cleanRange = cleanString(range);
        const computedFlag = calculateFlag(valNum, cleanRange, flag, testKey);

        const row = {
          test_name: testDisplayName,
          value: correctedVal,
          unit: correctedUnit,
          ref_range: cleanRange,
          flag: computedFlag,
          source_line: line,
        };
        row.plain_explanation = generatePlainExplanation(row);

        const dedupKey = `${testKey}_${row.value}`;
        if (!seen.has(dedupKey)) {
          seen.add(dedupKey);
          rows.push(row);
        }
        continue;
      }

      // 3. Clinical Dictionary Fallback for lines with irregular OCR spacing or typos
      const lowerLine = line.toLowerCase();
      for (const [dictKey, dictRef] of Object.entries(CLINICAL_REFERENCE_DB)) {
        let matchedIndex = lowerLine.indexOf(dictKey);

        // If direct match failed, try matching words in line with fuzzy distance
        if (matchedIndex === -1) {
          const words = lowerLine.split(/\s+/);
          for (let i = 0; i < words.length; i++) {
            const single = words[i];
            const pair = i + 1 < words.length ? `${words[i]} ${words[i + 1]}` : "";
            if (getSimilarity(single, dictKey) >= 0.82) {
              matchedIndex = lowerLine.indexOf(single);
              break;
            }
            if (pair && getSimilarity(pair, dictKey) >= 0.82) {
              matchedIndex = lowerLine.indexOf(pair);
              break;
            }
          }
        }

        if (matchedIndex !== -1) {
          const sub = line.substring(matchedIndex + dictKey.length);
          const numMatch = sub.match(/([0-9]+(?:[.,][0-9]+)?)/);
          if (numMatch) {
            const rawVal = numMatch[1];
            const correctedVal = repairValueDecimal(dictKey, rawVal);
            const valNum = parseFloat(correctedVal);

            const rangeMatch = sub.match(/([0-9]+(?:\.[0-9]+)?\s*[\-–—to]+\s*[0-9]+(?:\.[0-9]+)?)/);
            const refRange = rangeMatch ? rangeMatch[1] : `${dictRef.low} - ${dictRef.high} ${dictRef.unit}`;
            const computedFlag = calculateFlag(valNum, refRange, "", dictKey);

            const row = {
              test_name: dictRef.display,
              value: correctedVal,
              unit: dictRef.unit,
              ref_range: refRange,
              flag: computedFlag,
              source_line: line,
            };
            row.plain_explanation = generatePlainExplanation(row);

            const dedupKey = `${dictKey}_${correctedVal}`;
            if (!seen.has(dedupKey)) {
              seen.add(dedupKey);
              rows.push(row);
            }
            break;
          }
        }
      }
    }

    return rows;
  }

  // ---------------------------------------------------------------------------
  // Primary Public Parser: parseLabDocumentOnDevice
  // ---------------------------------------------------------------------------
  async function parseLabDocumentOnDevice(file, onProgress) {
    const notify = (pct, msg) => {
      if (typeof onProgress === "function") {
        onProgress({ percent: pct, message: msg });
      }
    };

    const isPdf = file.name.toLowerCase().endsWith(".pdf") || file.type === "application/pdf";
    notify(10, isPdf ? "Reading PDF in local browser memory…" : "Loading image into WebAssembly canvas…");

    const result = {
      size_bytes: file.size,
      ocr_used: false,
      ocr_available: true,
      client_side: true,
      raw_text: "",
      rows: [],
      warnings: [],
      error: null,
      disclaimer: CLINICAL_DISCLAIMER,
    };

    try {
      if (isPdf) {
        if (!window.pdfjsLib) {
          throw new Error("PDF.js library is not loaded. Please ensure script tag is present.");
        }

        if (!window.pdfjsLib.GlobalWorkerOptions.workerSrc) {
          window.pdfjsLib.GlobalWorkerOptions.workerSrc =
            "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
        }

        notify(25, "Extracting digital document text & layout…");
        const arrayBuffer = await file.arrayBuffer();
        const loadingTask = window.pdfjsLib.getDocument({ data: arrayBuffer });
        const pdf = await loadingTask.promise;

        let accumulatedLines = [];
        let totalRawText = "";
        let scannedPages = [];

        for (let p = 1; p <= pdf.numPages; p++) {
          notify(25 + Math.round((p / pdf.numPages) * 35), `Processing PDF page ${p} of ${pdf.numPages}…`);
          const page = await pdf.getPage(p);
          const textContent = await page.getTextContent();
          const pageLines = reconstructLinesFromPdfContent(textContent);

          const pageText = pageLines.join("\n");
          if (pageText.trim().length > 30) {
            accumulatedLines = accumulatedLines.concat(pageLines);
            totalRawText += (totalRawText ? "\n\n" : "") + pageText;
          } else {
            scannedPages.push({ pageNum: p, page });
          }
        }

        if (scannedPages.length > 0) {
          if (window.Tesseract) {
            notify(65, `Scanned page detected. Running On-Device Wasm OCR…`);
            result.ocr_used = true;
            for (const { pageNum, page } of scannedPages) {
              const viewport = page.getViewport({ scale: 2.0 });
              const canvas = document.createElement("canvas");
              canvas.width = viewport.width;
              canvas.height = viewport.height;
              const ctx = canvas.getContext("2d");
              await page.render({ canvasContext: ctx, viewport }).promise;

              preprocessImageCanvas(canvas);

              notify(75, `OCR analyzing scanned page ${pageNum}…`);
              const ocrRes = await window.Tesseract.recognize(canvas, "eng", {
                logger: m => {
                  if (m.status === "recognizing text") {
                    const ocrPct = Math.round((m.progress || 0) * 15);
                    notify(75 + ocrPct, `On-Device Wasm OCR (${Math.round((m.progress || 0) * 100)}%)…`);
                  }
                },
              });

              const ocrText = ocrRes?.data?.text || "";
              if (ocrText.trim()) {
                totalRawText += (totalRawText ? "\n\n" : "") + ocrText;
                accumulatedLines = accumulatedLines.concat(ocrText.split(/\r?\n/));
              }
            }
          } else {
            result.warnings.push(
              `Page(s) ${scannedPages.map(s => s.pageNum).join(", ")} contain scanned images. Enable Wasm OCR for full extraction.`
            );
          }
        }

        result.raw_text = totalRawText;
        notify(90, "Applying medical auto-correct & decimal repair…");
        result.rows = parseTextIntoStructuredRows(totalRawText);

      } else {
        if (!window.Tesseract) {
          throw new Error("Tesseract.js WebAssembly OCR engine is not loaded.");
        }

        notify(25, "Loading image into high-resolution canvas…");
        result.ocr_used = true;

        const img = new Image();
        const objUrl = URL.createObjectURL(file);

        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = () => reject(new Error("Unable to decode uploaded image in browser."));
          img.src = objUrl;
        });

        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth || img.width;
        canvas.height = img.naturalHeight || img.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(objUrl);

        notify(40, "Enhancing contrast and binarizing image…");
        preprocessImageCanvas(canvas);

        notify(50, "Running On-Device Wasm OCR…");
        const worker = await window.Tesseract.createWorker("eng", 1, {
          logger: m => {
            if (m.status === "recognizing text") {
              const p = Math.round((m.progress || 0) * 40);
              notify(50 + p, `On-Device Wasm OCR (${Math.round((m.progress || 0) * 100)}%)…`);
            }
          },
        });

        const ret = await worker.recognize(canvas);
        await worker.terminate();

        const fullText = ret?.data?.text || "";
        result.raw_text = fullText;

        notify(92, "Applying medical auto-correct & decimal repair…");
        result.rows = parseTextIntoStructuredRows(fullText);
      }

      if (!result.rows.length) {
        result.warnings.push(
          "No structured lab rows could be automatically detected from this document layout. " +
          "Please verify values directly from the raw extracted text or original document."
        );
      }

      notify(100, "Extraction complete ✓");
      return result;

    } catch (err) {
      console.error("[ClientParser] Error during on-device parsing:", err);
      result.error = err.message || "Failed to parse document in browser memory.";
      return result;
    }
  }

  // Exports
  window.parseLabDocumentOnDevice = parseLabDocumentOnDevice;
  window.CLINICAL_REFERENCE_DB = CLINICAL_REFERENCE_DB;
  window.parseTextIntoStructuredRows = parseTextIntoStructuredRows;
  window.repairValueDecimal = repairValueDecimal;
  window.fuzzyMatchTestName = fuzzyMatchTestName;

})(typeof window !== "undefined" ? window : (typeof global !== "undefined" ? global : this));
