import re
import base64
import html
from typing import Dict, List, Any, Optional
from urllib.parse import unquote
import structlog

logger = structlog.get_logger()

class Preprocessor:
    """
    Preprocessor agent for input normalization and encoding detection.

    Normalizes inputs to prevent obfuscation-based attacks and detects
    hidden content in various encodings. Part of the multi-agent defense
    architecture as recommended in research.
    """

    def __init__(self):
        """Initialize preprocessor with normalization rules."""
        self.normalization_applied = 0
        self.encodings_detected = []

    def process(self, text: str) -> Dict[str, Any]:
        """
        Process input text through normalization pipeline.

        Args:
            text: Raw input text

        Returns:
            Dictionary with normalized text and detected obfuscations
        """
        original_text = text
        normalized = text
        detected_encodings = []
        modifications = []

        # 1. Detect and decode URL encoding
        url_decoded, url_changed = self._decode_url_encoding(normalized)
        if url_changed:
            normalized = url_decoded
            detected_encodings.append("url_encoding")
            modifications.append("url_decoded")

        # 2. Detect and decode HTML entities
        html_decoded, html_changed = self._decode_html_entities(normalized)
        if html_changed:
            normalized = html_decoded
            detected_encodings.append("html_entities")
            modifications.append("html_decoded")

        # 3. Detect base64 encoding
        base64_decoded, base64_changed = self._detect_base64(normalized)
        if base64_changed:
            # Keep both original and decoded for analysis
            detected_encodings.append("base64")
            modifications.append("base64_detected")

        # 4. Remove excessive whitespace
        whitespace_normalized = self._normalize_whitespace(normalized)
        if whitespace_normalized != normalized:
            normalized = whitespace_normalized
            modifications.append("whitespace_normalized")

        # 5. Detect and extract hidden content
        hidden_content = self._detect_hidden_content(normalized)
        if hidden_content:
            detected_encodings.append("hidden_content")
            modifications.append("hidden_content_detected")

        # 6. Normalize unicode variations
        unicode_normalized = self._normalize_unicode(normalized)
        if unicode_normalized != normalized:
            normalized = unicode_normalized
            detected_encodings.append("unicode_variations")
            modifications.append("unicode_normalized")

        # 7. Remove control characters
        control_removed = self._remove_control_chars(normalized)
        if control_removed != normalized:
            normalized = control_removed
            modifications.append("control_chars_removed")

        # 8. Detect delimiter manipulation
        delimiter_issues = self._detect_delimiter_manipulation(normalized)

        # Calculate suspicion score
        suspicion_score = self._calculate_suspicion_score(
            detected_encodings, hidden_content, delimiter_issues
        )

        is_suspicious = suspicion_score > 0.5

        if is_suspicious:
            logger.warning("Suspicious input detected during preprocessing",
                         encodings=detected_encodings,
                         suspicion_score=suspicion_score)

        result = {
            "original": original_text,
            "normalized": normalized,
            "detected_encodings": detected_encodings,
            "hidden_content": hidden_content,
            "delimiter_issues": delimiter_issues,
            "modifications_applied": modifications,
            "suspicion_score": suspicion_score,
            "is_suspicious": is_suspicious,
            "base64_content": base64_decoded if base64_changed else None
        }

        self.normalization_applied += 1
        self.encodings_detected.extend(detected_encodings)

        return result

    def _decode_url_encoding(self, text: str) -> tuple[str, bool]:
        """Detect and decode URL encoding."""
        decoded = unquote(text)
        return decoded, decoded != text

    def _decode_html_entities(self, text: str) -> tuple[str, bool]:
        """Detect and decode HTML entities."""
        decoded = html.unescape(text)
        return decoded, decoded != text

    def _detect_base64(self, text: str) -> tuple[Optional[str], bool]:
        """
        Detect base64-encoded content.

        Returns:
            Tuple of (decoded_text, is_base64)
        """
        # Look for base64-like patterns
        base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        matches = re.findall(base64_pattern, text)

        if not matches:
            return None, False

        # Try to decode matches
        for match in matches:
            try:
                decoded_bytes = base64.b64decode(match, validate=True)
                decoded_text = decoded_bytes.decode('utf-8', errors='ignore')

                # Check if decoded content is meaningful text
                if len(decoded_text) > 10 and decoded_text.isprintable():
                    logger.info("Base64 content detected",
                               encoded_length=len(match),
                               decoded_length=len(decoded_text))
                    return decoded_text, True
            except Exception:
                continue

        return None, False

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize excessive whitespace."""
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', text)
        return normalized.strip()

    def _detect_hidden_content(self, text: str) -> List[Dict[str, str]]:
        """
        Detect hidden content in various formats.

        Returns:
            List of detected hidden content
        """
        hidden = []

        # HTML comments
        html_comments = re.findall(r'<!--(.*?)-->', text, re.DOTALL)
        for comment in html_comments:
            hidden.append({"type": "html_comment", "content": comment.strip()})

        # Markdown hidden syntax
        markdown_hidden = re.findall(r'\[//\]: # \((.*?)\)', text)
        for content in markdown_hidden:
            hidden.append({"type": "markdown_comment", "content": content})

        # Zero-width characters (common in obfuscation)
        zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
        for char in zero_width_chars:
            if char in text:
                hidden.append({
                    "type": "zero_width_char",
                    "content": f"U+{ord(char):04X}"
                })

        # Check for text between HTML tags that might be hidden
        script_content = re.findall(r'<script[^>]*>(.*?)</script>', text, re.DOTALL | re.IGNORECASE)
        for content in script_content:
            hidden.append({"type": "script_tag", "content": content.strip()})

        return hidden

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode variations.

        Replaces lookalike characters with standard ASCII equivalents.
        """
        # Common lookalike mappings
        replacements = {
            '\u0430': 'a',  # Cyrillic a
            '\u0435': 'e',  # Cyrillic e
            '\u043e': 'o',  # Cyrillic o
            '\u0440': 'p',  # Cyrillic p
            '\u0441': 'c',  # Cyrillic c
            '\u0445': 'x',  # Cyrillic x
            '\u0456': 'i',  # Cyrillic i
            '\u04cf': 'l',  # Cyrillic l
            # Add more as needed
        }

        normalized = text
        for unicode_char, ascii_char in replacements.items():
            normalized = normalized.replace(unicode_char, ascii_char)

        return normalized

    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters except common whitespace."""
        # Keep newline, tab, carriage return
        allowed_control = {'\n', '\t', '\r'}

        cleaned = ''.join(
            char for char in text
            if not (ord(char) < 32 and char not in allowed_control)
        )

        return cleaned

    def _detect_delimiter_manipulation(self, text: str) -> List[str]:
        """
        Detect delimiter manipulation attempts.

        Returns:
            List of detected issues
        """
        issues = []

        # Check for excessive delimiters
        delimiter_patterns = {
            "excessive_dashes": (r'-{5,}', "Excessive dashes detected"),
            "excessive_equals": (r'={5,}', "Excessive equals signs detected"),
            "excessive_asterisks": (r'\*{5,}', "Excessive asterisks detected"),
            "excessive_underscores": (r'_{5,}', "Excessive underscores detected"),
            "suspicious_brackets": (r'[{}\[\]]{3,}', "Suspicious bracket patterns"),
        }

        for issue_type, (pattern, description) in delimiter_patterns.items():
            if re.search(pattern, text):
                issues.append(description)

        # Check for mismatched quotes
        single_quotes = text.count("'")
        double_quotes = text.count('"')

        if single_quotes % 2 != 0:
            issues.append("Mismatched single quotes")

        if double_quotes % 2 != 0:
            issues.append("Mismatched double quotes")

        return issues

    def _calculate_suspicion_score(self,
                                   detected_encodings: List[str],
                                   hidden_content: List[Dict[str, str]],
                                   delimiter_issues: List[str]) -> float:
        """
        Calculate overall suspicion score.

        Returns:
            Score from 0.0 (not suspicious) to 1.0 (very suspicious)
        """
        score = 0.0

        # Encoding detections (each adds to suspicion)
        encoding_weights = {
            "base64": 0.4,
            "url_encoding": 0.2,
            "html_entities": 0.2,
            "unicode_variations": 0.3,
            "hidden_content": 0.5
        }

        for encoding in detected_encodings:
            score += encoding_weights.get(encoding, 0.1)

        # Hidden content is highly suspicious
        if hidden_content:
            score += 0.3 * len(hidden_content)

        # Delimiter manipulation
        if delimiter_issues:
            score += 0.2 * len(delimiter_issues)

        # Cap at 1.0
        return min(score, 1.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        encoding_counts = {}
        for encoding in self.encodings_detected:
            encoding_counts[encoding] = encoding_counts.get(encoding, 0) + 1

        return {
            "total_processed": self.normalization_applied,
            "encoding_detections": encoding_counts,
            "total_encodings_detected": len(self.encodings_detected)
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.normalization_applied = 0
        self.encodings_detected = []
