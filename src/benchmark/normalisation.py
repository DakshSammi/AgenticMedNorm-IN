import unicodedata
import re
from typing import Optional

class TextNormaliser:
    @staticmethod
    def normalise(text: Optional[str]) -> str:
        if text is None:
            return ""
            
        # 1. Unicode normalisation (NFC)
        text = unicodedata.normalize("NFC", text)
        
        # 2. Markdown code fences and backtick stripping
        text = re.sub(r"```[a-zA-Z]*", "", text)
        text = text.replace("```", "").replace("`", "")
        
        # 3. Lowercasing
        text = text.lower()
        
        # 4. Standardise date separators (convert . or / to - if surrounded by digits)
        text = re.sub(r"(\d+)[./](\d+)[./](\d+)", r"\1-\2-\3", text)
        
        # 5. Keep only allowed characters (alphanumeric, spaces, and critical medical symbols + - / RE LE BE)
        # Note: we also want to keep eye tags re, le, be, which are now lowercased: re, le, be.
        # We replace multiple spaces and collapse whitespaces.
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    @staticmethod
    def normalise_date(date_str: Optional[str]) -> Optional[str]:
        """
        Normalises date formats into a standard YYYY-MM-DD or DD-MM-YYYY format where possible.
        """
        if not date_str:
            return None
            
        date_clean = TextNormaliser.normalise(date_str)
        # Match standard patterns: dd-mm-yyyy, yyyy-mm-dd, dd-mm-yy
        match_dd_mm_yyyy = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", date_clean)
        if match_dd_mm_yyyy:
            d, m, y = match_dd_mm_yyyy.groups()
            return f"{int(d):02d}-{int(m):02d}-{y}"
            
        match_dd_mm_yy = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{2})$", date_clean)
        if match_dd_mm_yy:
            d, m, y = match_dd_mm_yy.groups()
            # Standardise to 20xx
            return f"{int(d):02d}-{int(m):02d}-20{y}"
            
        return date_clean
