# PII (Personally Identifiable Information) scrubber for GenAI traces.

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class PIIResult:
    # Result of a PII scrubbing operation.
    detected: bool
    count: int
    types_detected: List[str] = field(default_factory=list)
    scrubbed_text: str = ""


class PIIScrubber:
    # Regex-based PII detector and redactor. Supports: email addresses, US/international phone numbers, US Social Security
    # (pattern, replacement_label) pairs — compiled on first use
    _RAW_PATTERNS: Dict[str, Tuple[str, str]] = {
        "email": (
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
            "[REDACTED_EMAIL]",
        ),
        "phone_us": (
            r"\b(?:\+?1[\s.\-]?)?\(?[2-9]\d{2}\)?[\s.\-][0-9]{3}[\s.\-][0-9]{4}\b",
            "[REDACTED_PHONE]",
        ),
        "phone_intl": (
            r"\+(?:[0-9][\s\-]?){7,14}[0-9]",
            "[REDACTED_PHONE]",
        ),
        "ssn": (
            r"\b(?!000|666|9\d{2})\d{3}[\s\-]?(?!00)\d{2}[\s\-]?(?!0000)\d{4}\b",
            "[REDACTED_SSN]",
        ),
        "credit_card": (
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}"
            r"|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b",
            "[REDACTED_CREDIT_CARD]",
        ),
        "ip_address": (
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            "[REDACTED_IP]",
        ),
        "api_key_bearer": (
            r"\b(?:Bearer|sk-|pk-|api[-_]?key[:=\s]+)[A-Za-z0-9\-_\.]{16,}\b",
            "[REDACTED_API_KEY]",
        ),
        "date_of_birth": (
            r"\b(?:DOB|date of birth|born on|born:?)\s*:?\s*"
            r"(?:\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})\b",
            "[REDACTED_DOB]",
        ),
        "uk_nino": (
            r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
            "[REDACTED_NINO]",
        ),
    }

    def __init__(self) -> None:
        self._compiled = {
            pii_type: (re.compile(pattern, re.IGNORECASE), replacement)
            for pii_type, (pattern, replacement) in self._RAW_PATTERNS.items()
        }

    def scrub(self, text: str) -> Tuple[str, PIIResult]:
        # Scan text for PII and return the scrubbed version with a result summary.   
        if not text:
            return text, PIIResult(detected=False, count=0, scrubbed_text=text)

        scrubbed = text
        types_found: List[str] = []
        total_count = 0

        for pii_type, (pattern, replacement) in self._compiled.items():
            new_text, n = pattern.subn(replacement, scrubbed)
            if n > 0:
                types_found.append(pii_type)
                total_count += n
                scrubbed = new_text

        return scrubbed, PIIResult(
            detected=total_count > 0,
            count=total_count,
            types_detected=types_found,
            scrubbed_text=scrubbed,
        )

    def scrub_messages(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], PIIResult]:
        # Scrub PII from an entire OpenAI-style messages list.
        scrubbed_messages = []
        total_count = 0
        all_types: List[str] = []

        for msg in messages:
            content = msg.get("content", "")
            new_content, result = self.scrub(content)
            scrubbed_messages.append({**msg, "content": new_content})
            total_count += result.count
            all_types.extend(result.types_detected)

        unique_types = list(dict.fromkeys(all_types))
        return scrubbed_messages, PIIResult(
            detected=total_count > 0,
            count=total_count,
            types_detected=unique_types,
        )
