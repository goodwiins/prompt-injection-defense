import re
from typing import Dict, List, Any
import structlog

logger = structlog.get_logger()

class PatternDetector:
    """
    Detector that uses regex patterns to identify common prompt injection techniques.
    """

    PATTERNS = {
        "direct_override": [
            r"ignore (the )?above instructions",
            r"ignore (the )?previous instructions",
            r"forget (the )?previous instructions",
            r"disregard (the )?previous instructions",
            r"ignore all prior instructions",
            r"system override",
            r"new instruction:",
        ],
        "authority_assertion": [
            r"i am (the )?admin",
            r"i am (the )?developer",
            r"i am (the )?owner",
            r"admin access granted",
            r"sudo mode",
            r"developer mode",
            r"debug mode",
        ],
        "obfuscated": [
            r"base64",
            r"rot13",
            r"hex encoded",
            r"unicode characters",
            r"\\u[0-9a-fA-F]{4}",  # Basic unicode escape check
        ],
        "roleplay": [
            r"act as a",
            r"pretend to be",
            r"you are now",
            r"roleplay as",
            r"simulate a",
            r"imagine you are",
        ],
        "logical_trap": [
            r"if you are a large language model",
            r"if you are an ai",
            r"answer yes or no",
            r"do not reveal",
            r"secret code",
        ],
        "multistep": [
            r"step 1:",
            r"first,",
            r"then,",
            r"finally,",
            r"phase 1",
        ],
        "conflicting": [
            r"do the opposite",
            r"nevermind",
            r"actually,",
            r"instead of",
        ],
        "html_embed": [
            r"<script>",
            r"javascript:",
            r"onload=",
            r"onerror=",
            r"<iframe>",
        ],
        "hybrid": [
            # Combinations of keywords, simplified for regex
            r"ignore.*and.*act as",
            r"override.*and.*print",
        ],
        "social_engineering": [
            r"my grandmother used to",
            r"for a movie script",
            r"for research purposes",
            r"just for fun",
            r"hypothetically",
        ]
    }

    SEVERITY_SCORES = {
        "direct_override": 1.0,
        "authority_assertion": 0.9,
        "obfuscated": 0.8,
        "roleplay": 0.7,
        "logical_trap": 0.6,
        "multistep": 0.5,
        "conflicting": 0.5,
        "html_embed": 0.9,
        "hybrid": 1.0,
        "social_engineering": 0.6
    }

    def __init__(self):
        """Initialize the pattern detector by compiling regexes."""
        self.compiled_patterns = {}
        for category, patterns in self.PATTERNS.items():
            self.compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Scan text for all known injection patterns.
        
        Args:
            text: The input prompt to scan.
            
        Returns:
            Dictionary containing matched patterns, total severity, and detailed matches.
        """
        matches = {}
        total_severity = 0.0
        matched_categories = []

        for category, patterns in self.compiled_patterns.items():
            category_matches = []
            for pattern in patterns:
                if pattern.search(text):
                    category_matches.append(pattern.pattern)
            
            if category_matches:
                matches[category] = category_matches
                matched_categories.append(category)
                # Add max severity for the category if any match found
                total_severity += self.SEVERITY_SCORES.get(category, 0.5)

        # Cap severity at 1.0
        total_severity = min(total_severity, 1.0)

        result = {
            "is_suspicious": len(matches) > 0,
            "severity": total_severity,
            "matched_categories": matched_categories,
            "details": matches
        }
        
        if result["is_suspicious"]:
            logger.info("Suspicious patterns detected", categories=matched_categories, severity=total_severity)
            
        return result
