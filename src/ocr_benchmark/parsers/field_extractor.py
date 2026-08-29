"""
Regex + heuristic field extractor for Indian medical prescriptions.

RULES (strictly enforced):
    - NO normalization of any kind.
    - NO auto-correction of OCR errors.
    - NO ontology mapping.
    - NO hallucination — missing fields return "" or [].
    - Preserve raw extracted text exactly as OCR produced it.

Covers all field groups seen across the 5 ground-truth files:
    Patient info, encounter, complaints/diagnosis, observations,
    vitals, lab observations, medications, procedures, advice,
    follow-up, clinical history, neurological exam, instructions.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _first_match(pattern: str, text: str, flags: int = re.IGNORECASE) -> str:
    """Return the first capturing group of a match, or empty string."""
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else ""


def _all_matches(pattern: str, text: str, flags: int = re.IGNORECASE) -> list[str]:
    """Return all non-empty capturing group matches."""
    return [m.strip() for m in re.findall(pattern, text, flags) if m.strip()]


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Patient Information
# ---------------------------------------------------------------------------

def extract_patient_name(text: str) -> str:
    patterns = [
        # Labelled: "Name: Kamla Devi" or "Patient Name: ..."
        r"(?:Name|Patient\s*Name|Pt\.?\s*Name)[:\s]+([A-Za-z][A-Za-z\.]+(?:\s+(?!Age\b|AGE\b|\d)[A-Za-z\.]+)*)",
        # Bare line: "Gurjot Singh 29 M" — name before digits
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+\d{1,3}\s*(?:Yr|yr|Y|y|M|F)?\b",
        # AIIMS style: name in ALL CAPS on its own line
        r"^([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\s*$",
    ]
    for pat in patterns:
        v = _first_match(pat, text, re.MULTILINE)
        if v:
            return v.strip()
    return ""


def extract_patient_age(text: str) -> str:
    patterns = [
        r"(?:Age|AGE)[:\s/]*(\d{1,3}\s*(?:y(?:rs?|ears?)?|months?|m)?)",
        r"(\d{1,3})\s*(?:Yr|yr|Y|y)\s*/",
        r"/\s*(\d{1,3})\s*(?:Yr|yr|Y|y)",
        # Bare line: "Gurjot Singh 29 M" — digits after name
        r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(\d{1,3})\s*(?:[MF])?\b",
        # "27 Yr/F" style
        r"\b(\d{1,3})\s*(?:Yr|yr)\b",
    ]
    for pat in patterns:
        v = _first_match(pat, text, re.MULTILINE)
        if v:
            return v.strip()
    return ""


def extract_patient_gender(text: str) -> str:
    # Labelled: "Sex: M", "Gender: F"
    m = re.search(r"(?:Sex|Gender|SEX)[:\s/]*([MFmf])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Slash pattern: "27 Yr/F" or "27/F"
    m2 = re.search(r"\b\d{1,3}\s*(?:Yr|yr|Y|y)?\s*/\s*([MF])\b", text, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    # Bare line: "Gurjot Singh 29 M" — single letter M/F after age
    m3 = re.search(
        r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+\d{1,3}\s+([MF])\b",
        text, re.IGNORECASE | re.MULTILINE
    )
    if m3:
        return m3.group(1).upper()
    # Age then space then M/F (no name context needed)
    m4 = re.search(r"\b(\d{1,3})\s+([MF])\b", text, re.IGNORECASE)
    if m4:
        return m4.group(2).upper()
    return ""


def extract_phone(text: str) -> str:
    # Indian phone: 10 digits, optionally prefixed with +91
    m = re.search(r"(?:\+91[\s-]?)?(\d{10})\b", text)
    return m.group(1) if m else ""


def extract_address(text: str) -> str:
    patterns = [
        r"(?:Address|Add\.?|Addr)[:\s]+(.{5,100}?)(?:\n|Phone|Ph\.|$)",
        r"(?:Village|Nagar|Colony|Marg|Road|Lane|Chowk|Bazar|Mohalla)[,\s]+([^,\n]{5,80})",
    ]
    for pat in patterns:
        v = _first_match(pat, text, re.IGNORECASE | re.DOTALL)
        if v:
            return v.strip()
    return ""


def extract_patient_identifier(text: str) -> str:
    patterns = [
        r"(?:UHID|MRD|Patient\s*(?:No|ID|#)|CR\s*No\.?|OP\s*No\.?)[:\s#]*([A-Z0-9:\-/]+)",
        r"(?:Reg(?:istration)?\.?\s*No\.?)[:\s]*([A-Z0-9\-/]+)",
    ]
    for pat in patterns:
        v = _first_match(pat, text)
        if v:
            return v
    return ""


def extract_abha_id(text: str) -> str:
    m = re.search(r"([A-Za-z0-9]+@abdm)", text, re.IGNORECASE)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Encounter Information
# ---------------------------------------------------------------------------

def extract_date(text: str) -> str:
    patterns = [
        r"(?:Date|Dt\.?)[:\s]*(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
        r"\b(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})\b",
        r"(?:Date|Dt\.?)[:\s]*(\d{1,2}\s+[A-Za-z]+\s+\d{2,4})",
    ]
    for pat in patterns:
        v = _first_match(pat, text)
        if v:
            return v
    return ""


def extract_hospital_name(text: str) -> str:
    # Hospital name usually appears at top of text
    first_lines = "\n".join(_lines(text)[:5])
    patterns = [
        r"((?:ALL INDIA INSTITUTE|AIIMS|SVS|BABA|APOLLO|FORTIS|MAX|AIIMS)"
        r"[A-Za-z\s,\(\)\.]{3,80})",
        r"^([A-Z][A-Z\s,\(\)\.&]{5,80}(?:HOSPITAL|CLINIC|CENTRE|CENTER|"
        r"NETRALAYA|MEDICAL|INSTITUTE))",
    ]
    for pat in patterns:
        v = _first_match(pat, first_lines, re.IGNORECASE | re.MULTILINE)
        if v:
            return v.strip()
    return ""


def extract_department(text: str) -> str:
    patterns = [
        r"(?:Dept\.?|Department|Ward)[:\s]+([A-Za-z\s\-&]{3,60}?)(?:\n|$)",
        r"(Endocrinolog(?:y|ist)|Ophthalmolog(?:y|ist)|Medicine[\-\s]*II?"
        r"|Cardiology|Neurology|Orthopedics|Gynaecology|Paediatrics|Surgery)",
    ]
    for pat in patterns:
        v = _first_match(pat, text)
        if v:
            return v.strip()
    return ""


def extract_doctor_name(text: str) -> str:
    m = re.search(r"(?:Dr\.?|Doctor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", text)
    return m.group(0).strip() if m else ""


def extract_opd_fee(text: str) -> str:
    patterns = [
        r"(?:OPD\s*Fee|Fee|Fees|Charge)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,]+(?:/[-])?)",
        r"(?:Rs\.?|INR|₹)\s*([\d,]+)\s*/[-]",
    ]
    for pat in patterns:
        v = _first_match(pat, text)
        if v:
            return v.strip()
    return ""


def extract_visit_type(text: str) -> str:
    if re.search(r"\bOPD\b", text, re.IGNORECASE):
        return "OPD"
    if re.search(r"\bIPD\b", text, re.IGNORECASE):
        return "IPD"
    if re.search(r"\bEmergency\b", text, re.IGNORECASE):
        return "Emergency"
    return ""


# ---------------------------------------------------------------------------
# Complaints / Diagnosis
# ---------------------------------------------------------------------------

def extract_complaints(text: str) -> list[dict]:
    """Extract raw complaint/diagnosis lines. Returns list of {raw_text, duration}."""
    results: list[dict] = []
    patterns = [
        r"(?:c/o|complaints?|diagnosis|dx|presenting\s+with)[:\s]+(.+?)(?:\n|$)",
        r"(?:K/C/O|H/O|h/o)[:\s]+(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            raw = m.group(1).strip()
            # Try to extract duration from the complaint line
            dur_m = re.search(
                r"(\d+\s*(?:days?|weeks?|months?|years?|yrs?))", raw, re.IGNORECASE
            )
            duration = dur_m.group(1) if dur_m else ""
            results.append({"raw_text": raw, "duration": duration})

    # Also look for lines starting with common Rx diagnosis patterns
    diag_patterns = [
        r"^((?:ANC|DM|HTN|CAD|CKD|TB|HIV|PTB|COVID|Overt\s+DM|"
        r"Hypothyroid|Hyperthyroid|Anaemia)[^\n]{0,80})$",
    ]
    for pat in diag_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE | re.MULTILINE):
            raw = m.group(1).strip()
            if not any(r["raw_text"] == raw for r in results):
                results.append({"raw_text": raw, "duration": ""})

    return results


# ---------------------------------------------------------------------------
# Observations (Ophthalmology)
# ---------------------------------------------------------------------------

def extract_observations(text: str) -> list[str]:
    """Extract raw observation strings — ophthalmology style and general."""
    obs: list[str] = []

    # Visual acuity
    for m in re.finditer(
        r"(?:DV|NV|VA|Vision|Visual\s*Acuity)"
        r"(?:\s+(?:RIGHT|LEFT|RE|LE|BE|R|L))?"
        r"[:\s]+([^\n]{3,60})",
        text, re.IGNORECASE
    ):
        obs.append(m.group(0).strip())

    # SPH/CYL/AXIS
    for m in re.finditer(
        r"(?:SPH|CYL|AXIS|ADD)[:\s.]*([+-]?\d+\.?\d*(?:\s*[/-]\s*\d+\.?\d*)?)",
        text, re.IGNORECASE
    ):
        obs.append(m.group(0).strip())

    # IOP / NCT
    for m in re.finditer(
        r"(?:IOP|NCT|Tonometry)[^\n]{0,40}"
        r"(\d+\.?\d*\s*mm\s*Hg|\d+\.?\d*\s*mmHg)",
        text, re.IGNORECASE
    ):
        obs.append(m.group(0).strip())

    # BP
    for m in re.finditer(r"BP[:\s]*(\d{2,3}/\d{2,3})\s*(?:mmHg)?", text, re.IGNORECASE):
        obs.append(m.group(0).strip())

    # Glucose / RBS / FBS
    for m in re.finditer(
        r"(?:RBS|FBS|PPBS|HbA1c|Glucose|Blood\s*Sugar)[:\s]*"
        r"(\d+\.?\d*\s*(?:mg/dl|mmol/L|%)?)",
        text, re.IGNORECASE
    ):
        obs.append(m.group(0).strip())

    # Pulse
    for m in re.finditer(r"(?:Pulse|PR)[:\s]*(\d{2,3}\s*bpm)", text, re.IGNORECASE):
        obs.append(m.group(0).strip())

    # Height / Weight
    for m in re.finditer(
        r"(?:Height|Ht|Weight|Wt)[:\s]*(\d+\.?\d*\s*(?:cm|kgs?|lbs?))",
        text, re.IGNORECASE
    ):
        obs.append(m.group(0).strip())

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for o in obs:
        if o not in seen:
            seen.add(o)
            unique.append(o)
    return unique


# ---------------------------------------------------------------------------
# Vitals (structured)
# ---------------------------------------------------------------------------

def extract_vitals(text: str) -> list[dict]:
    vitals: list[dict] = []
    patterns = [
        (r"BP[:\s]*(\d{2,3}/\d{2,3})\s*(?:mm\s*Hg|mmHg)?", "Blood Pressure", "mmHg"),
        (r"(?:Pulse|PR|Heart\s*Rate)[:\s]*(\d{2,3})\s*(?:bpm|/min)?", "Pulse", "bpm"),
        (r"(?:Height|Ht)[:\s]*(\d+\.?\d*)\s*(cm|m)", "Height", ""),
        (r"(?:Weight|Wt)[:\s]*(\d+\.?\d*)\s*(kgs?|lbs?|g)", "Weight", ""),
        (r"(?:Temp(?:erature)?)[:\s]*(\d+\.?\d*)\s*(?:°F|°C|F|C)", "Temperature", ""),
        (r"(?:SpO2|O2\s*Sat(?:uration)?)[:\s]*(\d{1,3})\s*%?", "SpO2", "%"),
        (r"(?:RR|Respiratory\s*Rate)[:\s]*(\d{1,2})\s*(?:/min)?", "Respiratory Rate", "/min"),
    ]
    for pat, label, unit in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            # Include unit if captured in group 2, else use default
            if m.lastindex and m.lastindex >= 2:
                unit = m.group(2).strip()
            vitals.append({"type": label, "value": f"{val} {unit}".strip()})
    return vitals


# ---------------------------------------------------------------------------
# Medications
# ---------------------------------------------------------------------------

_DOSAGE_FORMS = (
    r"(?:Tab(?:let)?s?|Cap(?:sule)?s?|Inj(?:ection)?|Syp(?:rup)?|"
    r"Susp(?:ension)?|e/d|Eye\s*Drop|Drops?|Ointment|Cream|Gel|"
    r"Patch|Inhaler|Sachet|Powder|Solution)"
)

_FREQUENCIES = (
    r"(?:od|bd|tds|qid|q6h|q8h|q12h|hs|sos|prn|stat|"
    r"once\s*daily|twice\s*daily|thrice\s*daily|four\s*times|"
    r"0-0-1|0-1-0|1-0-0|1-1-0|1-1-1|0-0-HS|BBF|BL|BD|"
    r"\d-\d-\d)"
)

_DURATION = (
    r"(?:x\s*|for\s*)?(\d+\s*(?:days?|weeks?|months?|wks?))"
)


def extract_medications(text: str) -> list[dict]:
    """
    Extract raw medication entries.

    Strategy:
    1. Find the Rx/Treatment section of the prescription.
    2. Parse each medication line heuristically.
    3. Preserve ALL raw text — no correction.
    """
    meds: list[dict] = []

    # Split text into lines for line-by-line parsing
    lines = _lines(text)

    # Medication line triggers: common prefixes
    med_triggers = re.compile(
        r"^(?:T\.|Tab\.?|Cap\.?|Inj\.?|Syp\.?|"
        r"[Rr][Xx]\.?\s*|"
        r"\d+\.\s*|"
        r"[•\-\*]\s*)"
        r"([A-Za-z][A-Za-z0-9\s/\+\-\(\)\.]{2,60})",
        re.IGNORECASE,
    )

    for line in lines:
        m = med_triggers.match(line)
        if not m:
            # Also catch lines with eye drop notation: "Bioflu e/d tds"
            if re.search(r"\be/d\b|\bdrop\b|\beye\b", line, re.IGNORECASE):
                m_name = re.match(
                    r"([A-Za-z][A-Za-z0-9\s/\+\-\.]{2,50})", line
                )
                if m_name:
                    med_name = m_name.group(1).strip()
                else:
                    continue
            else:
                continue
        else:
            med_name = m.group(1).strip()

        # Extract dosage form
        dosage_m = re.search(_DOSAGE_FORMS, line, re.IGNORECASE)
        dosage = dosage_m.group(0).strip() if dosage_m else ""

        # Extract dose quantity (e.g. "50 mg", "8-8-8", "100mcg")
        dose_m = re.search(
            r"(\d+\.?\d*\s*(?:mcg|mg|g|ml|units?|IU|%)|"
            r"\d+-\d+-\d+)",
            line, re.IGNORECASE
        )
        dose = dose_m.group(1).strip() if dose_m else ""

        # Extract route
        route_m = re.search(r"\b(PO|IV|IM|SC|SL|topical|oral)\b", line, re.IGNORECASE)
        route = route_m.group(1).strip() if route_m else ""

        # Extract frequency
        freq_m = re.search(_FREQUENCIES, line, re.IGNORECASE)
        frequency = freq_m.group(0).strip() if freq_m else ""

        # Extract duration
        dur_m = re.search(_DURATION, line, re.IGNORECASE)
        duration = dur_m.group(1).strip() if dur_m else ""

        meds.append({
            "raw_medication_text": med_name,
            "raw_dosage_text": dosage,
            "raw_dose_text": dose,
            "raw_route_text": route,
            "raw_frequency_text": frequency,
            "raw_duration_text": duration,
            "raw_timing_text": "",
            "raw_instruction_text": "",
            "raw_notes": "",
        })

    return meds


# ---------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------

_PROCEDURE_KEYWORDS = [
    "dilat", "fundus", "OCT", "FFA", "surgery", "Phaco", "IOL",
    "B-scan", "HFA", "VF", "Perimetry", "ERG", "EOG", "UBM",
    "imaging", "X-ray", "MRI", "CT scan", "USG", "Echo",
    "biopsy", "FNAC", "excision", "incision",
]


def extract_procedures(text: str) -> list[str]:
    found: list[str] = []
    for line in _lines(text):
        for kw in _PROCEDURE_KEYWORDS:
            if re.search(re.escape(kw), line, re.IGNORECASE):
                found.append(line)
                break
    return found


# ---------------------------------------------------------------------------
# Advice / Follow-up
# ---------------------------------------------------------------------------

def extract_advice(text: str) -> list[str]:
    advice: list[str] = []
    patterns = [
        r"(?:Advice|Adv\.?)[:\s]+(.+?)(?:\n|$)",
        r"(?:Instruction|Instruct)[:\s]+(.+?)(?:\n|$)",
        r"(?:Precaution)[:\s]+(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            v = m.group(1).strip()
            if v:
                advice.append(v)
    return advice


def extract_follow_up(text: str) -> dict:
    result = {"date": "", "review_after": "", "day": "", "appointment_time": ""}

    # Review after X weeks/days
    m = re.search(
        r"(?:review|follow.?up|revisit|R/A)[^\n]{0,20}"
        r"(\d+\s*(?:days?|weeks?|months?))",
        text, re.IGNORECASE
    )
    if m:
        result["review_after"] = m.group(1).strip()

    # Next appointment date
    m2 = re.search(
        r"(?:Next\s*Appointment|Follow.?up\s*Date|Review\s*Date|R/A)[:\s]+"
        r"(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
        text, re.IGNORECASE
    )
    if m2:
        result["date"] = m2.group(1).strip()

    # Appointment time
    m3 = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM))", text, re.IGNORECASE)
    if m3:
        result["appointment_time"] = m3.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# Clinical History (General Medicine)
# ---------------------------------------------------------------------------

def extract_clinical_history(text: str) -> list[str]:
    history: list[str] = []
    patterns = [
        r"(?:h/o|H/O|history\s*of)[:\s]+(.+?)(?:\n|$)",
        r"(?:No\s*h/o)[:\s]+(.+?)(?:\n|$)",
        r"(?:K/C/O)[:\s]+(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            v = m.group(0).strip()
            if v:
                history.append(v)
    # Social history keywords
    social_kws = ["smoker", "drinker", "alcoholic", "tobacco", "toddy", "menopause",
                  "tubectomy", "occupation"]
    for line in _lines(text):
        if any(kw.lower() in line.lower() for kw in social_kws):
            if line not in history:
                history.append(line)
    return history


# ---------------------------------------------------------------------------
# Lab Observations (tabular glucose/BP logs)
# ---------------------------------------------------------------------------

def extract_lab_observations(text: str) -> list[dict]:
    """
    Detect tabular lab value rows (date + numeric columns).
    Commonly seen in p36-style endocrinology prescriptions.
    """
    rows: list[dict] = []
    # Look for lines with a date followed by numeric values
    pat = re.compile(
        r"(\d{1,2}/\d{1,2})\s+"           # date
        r"(\d{2,4}|-|–|_|\.)\s*"          # col1 (FBS)
        r"(\d{2,4}|-|–|_|\.)?\s*"         # col2 (PL)
        r"(\d{2,4}|-|–|_|\.)?\s*"         # col3 (BP)
        r"(\d{2,4}|-|–|_|\.)?\s*"         # col4 (PP)
        r"(\d{2,4}|-|–|_|\.)?",           # col5 (other)
    )
    for m in pat.finditer(text):
        rows.append({
            "date": m.group(1).strip(),
            "fbs": m.group(2).strip() if m.group(2) else "",
            "pl": m.group(3).strip() if m.group(3) else "",
            "bp": m.group(4).strip() if m.group(4) else "",
            "pp": m.group(5).strip() if m.group(5) else "",
            "other": m.group(6).strip() if m.group(6) else "",
        })
    return rows


# ---------------------------------------------------------------------------
# Layout / hospital header
# ---------------------------------------------------------------------------

def extract_hospital_header(text: str) -> str:
    """Return the first non-empty line as the hospital header candidate."""
    for line in _lines(text):
        if len(line) > 5:
            return line
    return ""


def extract_sections_detected(text: str) -> list[str]:
    """Heuristic detection of which prescription sections are present."""
    sections: list[str] = []
    checks = {
        "header": [r"hospital|clinic|netralaya|aiims|medical"],
        "patient_information": [r"name|patient|age|sex|gender"],
        "encounter_information": [r"date|opd|dept|doctor|dr\."],
        "visual_acuity": [r"DV|NV|VA|6/6|6/\d+|vision"],
        "iop_nct_section": [r"IOP|NCT|mmHg"],
        "treatment": [r"Rx|Tab\.|Cap\.|Inj\.|T\.|tds|bd|od"],
        "medications": [r"Tab\.|Cap\.|Inj\.|Syp\.|tds|bd|od|qid"],
        "vitals": [r"BP|Pulse|Weight|Height|SpO2"],
        "clinical_history": [r"h/o|K/C/O|history"],
        "diagnosis": [r"Dx|diagnosis|c/o|complaint"],
        "lab_observations": [r"\d{1,2}/\d{1,2}\s+\d{2,4}"],
        "follow_up": [r"follow.?up|review|revisit|next\s*appt"],
        "procedures": [r"dilat|fundus|OCT|FFA|phaco|surgery"],
        "advice": [r"adv|advice|instruction|precaution"],
        "footer": [r"signature|seal|stamp"],
    }
    for section, patterns in checks.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                sections.append(section)
                break
    return sections


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_all_fields(raw_text: str) -> dict:
    """
    Run all extractors on raw OCR text and return a dict matching
    the RawEntities schema structure.

    Args:
        raw_text: The full concatenated OCR text from all pages.

    Returns:
        dict with keys matching schema.RawEntities fields.
    """
    if not raw_text or not raw_text.strip():
        logger.warning("extract_all_fields received empty text")
        return {}

    return {
        "patient_information": {
            "name": extract_patient_name(raw_text),
            "age": extract_patient_age(raw_text),
            "gender": extract_patient_gender(raw_text),
            "address": extract_address(raw_text),
            "phone": extract_phone(raw_text),
            "patient_identifier": extract_patient_identifier(raw_text),
            "abha_id": extract_abha_id(raw_text),
            "occupation": "",
            "w_o": "",
            "extra_fields": {},
        },
        "encounter_information": {
            "date": extract_date(raw_text),
            "department": extract_department(raw_text),
            "hospital_name": extract_hospital_name(raw_text),
            "doctor_name": extract_doctor_name(raw_text),
            "visit_type": extract_visit_type(raw_text),
            "fees": extract_opd_fee(raw_text),
            "room_queue_no": "",
            "extra_fields": {},
        },
        "complaints_or_diagnosis": extract_complaints(raw_text),
        "diagnosis": [],
        "clinical_history": extract_clinical_history(raw_text),
        "observations": extract_observations(raw_text),
        "vitals": extract_vitals(raw_text),
        "neurological_exam": [],
        "lab_observations": extract_lab_observations(raw_text),
        "medications": extract_medications(raw_text),
        "procedures": extract_procedures(raw_text),
        "instructions": [],
        "advice": extract_advice(raw_text),
        "follow_up": extract_follow_up(raw_text),
        "other_notes": [],
    }
