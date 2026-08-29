"""
Common Indian medical abbreviations and their normalized expansions.
"""

# Frequency / Timing
FREQUENCY_MAP = {
    "od": "once daily",
    "bd": "twice daily",
    "tds": "three times daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "q6h": "every 6 hours",
    "q8h": "every 8 hours",
    "q12h": "every 12 hours",
    "hs": "at bedtime",
    "sos": "as needed",
    "prn": "as needed",
    "stat": "immediately",
    "bbf": "before breakfast",
    "af": "after food",
    "pc": "after meals",
    "ac": "before meals",
}

# Dosage Forms
DOSAGE_FORM_MAP = {
    "tab": "tablet",
    "tabs": "tablets",
    "cap": "capsule",
    "caps": "capsules",
    "inj": "injection",
    "syp": "syrup",
    "susp": "suspension",
    "e/d": "eye drops",
    "ed": "eye drops",
    "oint": "ointment",
    "crm": "cream",
    "gtte": "drops",
}

# Routes
ROUTE_MAP = {
    "po": "oral",
    "iv": "intravenous",
    "im": "intramuscular",
    "sc": "subcutaneous",
    "sl": "sublingual",
    "it": "intrathecal",
    "id": "intradermal",
}

# Common Diseases / Conditions
CONDITION_MAP = {
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "cad": "coronary artery disease",
    "ckd": "chronic kidney disease",
    "ptb": "pulmonary tuberculosis",
    "hypo": "hypothyroidism",
    "hyper": "hyperthyroidism",
    "anc": "antenatal care",
    "all.": "allergic",
    "conj.": "conjunctivitis",
}

# Merge all for quick lookup
MEDICAL_ABBREVIATIONS = {**FREQUENCY_MAP, **DOSAGE_FORM_MAP, **ROUTE_MAP, **CONDITION_MAP}
