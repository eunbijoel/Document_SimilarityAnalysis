"""텍스트 정제 및 문장 분리 유틸리티."""
import re

# 마침표/물음표/느낌표 + 보고서 개조식(ㅇ, -) 경계를 함께 고려
# 초안 pdf_similarity_checker.py 의 split_sentences 규칙을 반영
_SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?。！？])\s+|(?=ㅇ\s)|(?=\-\s)|(?=–\s)|(?=•\s)|\n+"
)
_WHITESPACE_PATTERN = re.compile(r"\s+")
_ONLY_DIGITS_PATTERN = re.compile(r"^[\d\s\.\,\-\%\(\)]+$")


def normalize_whitespace(text: str) -> str:
    """불필요한 공백과 줄바꿈을 하나의 공백으로 정리합니다. 원문 자체는 건드리지 않고
    비교용 정규화 문자열을 새로 만들 때 사용합니다."""
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def split_into_sentences(raw_text: str) -> list[str]:
    """페이지/슬라이드 단위 텍스트를 문장 단위로 분리합니다.

    한국어 보고서의 개조식(ㅇ, -, • 등)도 분리 기준으로 포함합니다.
    """
    if not raw_text:
        return []
    candidates = _SENTENCE_SPLIT_PATTERN.split(raw_text)
    return [c.strip() for c in candidates if c and c.strip()]


def is_valid_sentence(text: str, min_length: int = 8) -> bool:
    """너무 짧거나 숫자/기호로만 이루어진 문장을 걸러냅니다."""
    stripped = text.strip()
    if len(stripped) < min_length:
        return False
    if _ONLY_DIGITS_PATTERN.match(stripped):
        return False
    # 한글/영문 알파벳이 하나도 없으면 제외 (표, 목차 번호 등)
    if not re.search(r"[가-힣a-zA-Z]", stripped):
        return False
    return True


def make_sentence_id(file_name: str, index: int) -> str:
    return f"{file_name}::{index}"
