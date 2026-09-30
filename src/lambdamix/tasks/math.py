"""Shared math-answer equivalence checker."""

from __future__ import annotations


def _normalize_answer(answer: object) -> str:
    """Normalize answer-like values for comparison."""
    answer = str(answer).strip().replace(",", "")
    # Remove trailing period
    if answer.endswith("."):
        answer = answer[:-1]
    return answer


_MATH_VERIFY_IMPORT_WARNED = False


def math_verify_equal(prediction: object, reference: object) -> bool:
    """Check answer correctness using math-verify with exact-match fallback."""
    pred_norm = _normalize_answer(prediction)
    ref_norm = _normalize_answer(reference)

    # Exact match
    if pred_norm == ref_norm:
        return True

    try:
        from math_verify import parse, verify

        ref_parsed = parse(f"${str(reference).strip()}$")
        pred_parsed = parse(f"${str(prediction).strip()}$")
        if ref_parsed and pred_parsed and verify(ref_parsed, pred_parsed):
            return True
    except ImportError:
        global _MATH_VERIFY_IMPORT_WARNED
        if not _MATH_VERIFY_IMPORT_WARNED:
            _MATH_VERIFY_IMPORT_WARNED = True
            print(
                "WARNING: math_verify is not installed; math answer matching "
                "degrades to exact string/float comparison."
            )
    except Exception:
        pass

    # Numeric comparison fallback
    try:
        return float(pred_norm) == float(ref_norm)
    except (ValueError, TypeError):
        return False
