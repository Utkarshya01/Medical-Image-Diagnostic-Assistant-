const fs = require('fs');
eval(fs.readFileSync('frontend/client_parser.js', 'utf8'));

const messyOcrText = `
CLINICAL BIOCHEMISTRY & HAEMATOLOGY REPORT

Investigation               Observed Value   Unit     Reference Range   Status
Hemogiobin                  145              g/dI     13.0 - 17.0       Normal
Plateiets                   250              10*3/ul  150 - 450         Normal
Creatlnine                  09               mg/di    0.7 - 1.3         Normal
Giucose                     9.5              mg/di    70 - 99           Normal
Total CholesteroI           1.85             mg/dl    0 - 200           Normal
HbA1c                       58               %        4.0 - 5.6         Hlgh
RBC Count                   48               10^6/ul  3.8 - 5.8         Normal
Serum Caicium               92               mg/dl    8.5 - 10.5        Normal
`;

const rows = parseTextIntoStructuredRows(messyOcrText);

console.log(`Parsed ${rows.length} rows successfully:\n`);
rows.forEach((r, idx) => {
  console.log(`[${idx + 1}] Test Name: '${r.test_name}'`);
  console.log(`    Value: '${r.value}' (Repaired decimal / integer)`);
  console.log(`    Unit: '${r.unit}' | Range: '${r.ref_range}' | Flag: '${r.flag}'`);
  console.log(`    Explanation: ${r.plain_explanation}\n`);
});

