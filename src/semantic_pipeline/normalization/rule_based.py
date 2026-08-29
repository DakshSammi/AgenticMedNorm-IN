"""
Rule-based normalization for clinical text.
"""

import re
from .medical_terms import MEDICAL_ABBREVIATIONS

class RuleBasedNormalizer:
    def __init__(self):
        # Compile patterns for efficiency
        self.abbrev_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in MEDICAL_ABBREVIATIONS.keys()) + r')\b',
            re.IGNORECASE
        )

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        
        # 1. Basic cleaning
        text = text.strip()
        
        # 2. Expand abbreviations
        def replace_abbrev(match):
            abbrev = match.group(1).lower()
            return MEDICAL_ABBREVIATIONS.get(abbrev, match.group(1))
        
        normalized = self.abbrev_pattern.sub(replace_abbrev, text)
        
        # 3. Clean up punctuation/spacing
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'\.{2,}', '.', normalized)
        
        return normalized

    def normalize_medication(self, med_dict: dict) -> dict:
        """Normalizes fields in a medication dictionary."""
        normalized_med = med_dict.copy()
        
        for key in med_dict:
            if key.startswith("raw_"):
                norm_key = key.replace("raw_", "normalized_")
                normalized_med[norm_key] = self.normalize(str(med_dict[key]))
        
        return normalized_med
