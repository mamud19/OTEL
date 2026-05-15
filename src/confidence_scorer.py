# Confidence and uncertainty scorer for GenAI responses.
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ScoringResult:
    # Result of scoring a single LLM response.
    confidence_score: float
    # Confidence that the response is accurate; range [0.0, 1.0].

    UNCERTAINTY_RISK: float
    # Estimated probability the response contains hallucinated content; range [0.0, 1.0].

    risk_level: str
    # Categorical risk: low, medium, or high

    uncertainty_count: int
    # Total number of uncertainty/hedging markers found.

    uncertainty_markers: List[str] = field(default_factory=list)
    # Sample of the marker patterns that were matched (up to 10).


class ConfidenceScorer:
    """
    Scoring algorithm
    -----------------
    1. Count *low-confidence* markers (strong epistemic uncertainty):
       deduct 0.12 per match, capped at 0.60.
    2. Count *medium-hedging* markers (soft hedging language):
       deduct 0.04 per match, capped at 0.20.
    3. confidence = max(0.0, 1.0 - low_deduction - medium_deduction)
    4. UNCERTAINTY_RISK = 1.0 - confidence
    5. risk_level: < 0.30 → "low", < 0.60 → "medium", else → "high"
    """

    # Strong epistemic uncertainty each hit signals the model doesn't know
    LOW_CONFIDENCE_MARKERS: List[str] = [
        r"i'?m not sure",
        r"i don'?t know",
        r"i'?m uncertain",
        r"i'?m unsure",
        r"not certain",
        r"i can'?t be sure",
        r"i cannot be certain",
        r"i'?m not confident",
        r"without certainty",
        r"i have no way (?:of knowing|to know)",
        r"i lack (?:the )?(?:information|knowledge|data|context)",
        r"i don'?t have (?:access|information|data|the (?:ability|knowledge))",
        r"i cannot (?:confirm|verify|guarantee|determine|ascertain)",
        r"unverified",
        r"unconfirmed",
        r"i'?m (?:just )?(?:making|guessing|speculating)",
        r"\bspeculat(?:e|ion|ive|ing)\b",
        r"my (?:training )?(?:data|knowledge|information) (?:may|might) be (?:outdated|limited|incomplete)",
        r"as of my (?:knowledge |training )?cut-?off",
        r"i don'?t have (?:real-?time|current|live|up-?to-?date)",
        r"this (?:may|might|could) (?:not )?be (?:accurate|correct|true)",
    ]

    # Softer hedging the model is less than fully confident
    MEDIUM_HEDGING_MARKERS: List[str] = [
        r"\bpossibly\b",
        r"\bperhaps\b",
        r"\bmaybe\b",
        r"\bmight\b",
        r"\bcould be\b",
        r"\bi think\b",
        r"\bi believe\b",
        r"\bi assume\b",
        r"\bit seems\b",
        r"\bapparently\b",
        r"\ballegedly\b",
        r"\bsupposedly\b",
        r"\blikely\b",
        r"\bprobably\b",
        r"\bseems (?:like|to be)\b",
        r"\bappears (?:to be)?\b",
        r"\btypically\b",
        r"\bgenerally\b",
        r"\busually\b",
        r"\bin most cases\b",
        r"\bto my (?:knowledge|understanding|recollection)\b",
    ]

    def __init__(self) -> None:
        self._low_re = [re.compile(p, re.IGNORECASE) for p in self.LOW_CONFIDENCE_MARKERS]
        self._med_re = [re.compile(p, re.IGNORECASE) for p in self.MEDIUM_HEDGING_MARKERS]

    def score(self, text: str) -> ScoringResult:
        # Score a single LLM response string.
        if not text:
            return ScoringResult(
                confidence_score=1.0,
                UNCERTAINTY_RISK=0.0,
                risk_level="low",
                uncertainty_count=0,
                uncertainty_markers=[],
            )

        text_lower = text.lower()
        matched_labels: List[str] = []

        low_count = 0
        for i, pattern in enumerate(self._low_re):
            if pattern.search(text_lower):
                low_count += 1
                matched_labels.append(self.LOW_CONFIDENCE_MARKERS[i][:40])

        medium_count = 0
        for i, pattern in enumerate(self._med_re):
            if pattern.search(text_lower):
                medium_count += 1
                if len(matched_labels) < 10:
                    matched_labels.append(self.MEDIUM_HEDGING_MARKERS[i][:40])

        low_deduction = min(low_count * 0.12, 0.60)
        medium_deduction = min(medium_count * 0.04, 0.20)
        confidence = round(max(0.0, 1.0 - low_deduction - medium_deduction), 4)
        risk = round(1.0 - confidence, 4)

        if risk < 0.30:
            risk_level = "low"
        elif risk < 0.60:
            risk_level = "medium"
        else:
            risk_level = "high"

        return ScoringResult(
            confidence_score=confidence,
            UNCERTAINTY_RISK=risk,
            risk_level=risk_level,
            uncertainty_count=low_count + medium_count,
            uncertainty_markers=matched_labels[:10],
        )

    def score_batch(self, texts: List[str]) -> List[ScoringResult]:
        # Score a list of completion strings in sequence.
        return [self.score(t) for t in texts]
