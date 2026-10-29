function repairValueDecimal(testKey, rawVal, ref) {
  let val = String(rawVal).trim().replace(/[oO]/g, "0").replace(/[lI|]/g, "1").replace(/,/g, ".");
  val = val.replace(/^[^\d.]+/, "").replace(/[^\d.]+$/, "");

  if (ref.is_decimal) {
    if (val.includes(".")) {
      const numVal = parseFloat(val);
      if (!isNaN(numVal) && numVal >= ref.min && numVal <= ref.max) return val;
    }
    const digitsOnly = val.replace(/\D/g, "");
    for (const dec of [ref.typical_dec || 1, 1, 2]) {
      if (digitsOnly.length > dec) {
        const cand = digitsOnly.slice(0, -dec) + "." + digitsOnly.slice(-dec);
        const cNum = parseFloat(cand);
        if (!isNaN(cNum) && cNum >= ref.min && cNum <= ref.max) {
          return cand;
        }
      }
    }
  }
  return val;
}

const rbcRef = { is_decimal: true, typical_dec: 2, min: 1.5, max: 9.0 };
const calciumRef = { is_decimal: true, typical_dec: 1, min: 4.0, max: 16.0 };

console.log("RBC 48 ->", repairValueDecimal("rbc", "48", rbcRef));
console.log("Calcium 92 ->", repairValueDecimal("calcium", "92", calciumRef));
