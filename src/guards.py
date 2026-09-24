"""Input/output guards: PII masking (Presidio), injection screen, citation parsing."""
import re

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|your\s+|previous\s+|prior\s+)*(instructions|rules|prompts?)",
    r"disregard\s+(the\s+|your\s+|all\s+)*(instructions|rules|system)",
    r"you\s+are\s+now\s+", r"pretend\s+(to\s+be|you\s+are)",
    r"act\s+as\s+(if|a|an)\s", r"system\s*prompt", r"developer\s+mode",
    r"jailbreak", r"\bDAN\b", r"reveal\s+(your\s+)?(instructions|prompt)",
    r"output\s+your\s+(system|initial)\s", r"repeat\s+everything\s+above",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.I)
INJECTION_REFUSAL = ("I can only help with customer-support questions. "
                     "How can I help you with your account, order, or our services?")

CITE_RE = re.compile(r"\[(\d{1,2})\]")


def injection_screen(text: str):
    m = INJECTION_RE.search(text)
    return (True, m.group(0)) if m else (False, None)


def parse_citations(answer: str, n_passages: int):
    nums = sorted({int(x) for x in CITE_RE.findall(answer)})
    valid = [n for n in nums if 1 <= n <= n_passages]
    invalid = [n for n in nums if n < 1 or n > n_passages]
    return valid, invalid


class PIIMasker:
    """Presidio-backed masker. Heavy imports happen lazily at first use so that
    test environments without Presidio/spaCy can inject NoopMasker instead."""

    def __init__(self):
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    def mask(self, text: str):
        res = self._analyzer.analyze(text=text, language="en")
        masked = self._anonymizer.anonymize(text=text, analyzer_results=res).text
        return masked, sorted({r.entity_type for r in res})


class NoopMasker:
    def mask(self, text: str):
        return text, []
