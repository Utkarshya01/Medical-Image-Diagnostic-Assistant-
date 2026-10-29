const fs = require('fs');
eval(fs.readFileSync('frontend/client_parser.js', 'utf8'));

const testText = `
COMPLETE BLOOD COUNT (CBC)
Investigation               Observed Value   Unit       Reference Interval   Status
Hemoglobin                  14.5             g/dL       13.0 - 17.0          Normal
Total Leucocyte Count (TLC) 12500            10^3/uL    4000 - 11000         High
Platelet Count              95               10^3/uL    150 - 450            Low
Fasting Blood Sugar: 115 mg/dL (Ref: 70 - 99) H
Serum Creatinine 0.9 mg/dL 0.7 - 1.3
`;

const rows = parseTextIntoStructuredRows(testText);
console.log('Extracted count:', rows.length);
rows.forEach((r, idx) => {
  console.log(`[${idx + 1}] ${r.test_name}: ${r.value} ${r.unit} | Range: ${r.ref_range} | Flag: ${r.flag}`);
  console.log(`    Explanation: ${r.plain_explanation}`);
});

