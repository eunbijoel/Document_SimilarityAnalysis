"""양식/공통 문장 필터.

보고서 서식·기관명·표 라벨처럼 '내용 재사용'이 아닌 문장을
유사 문장 비교에서 제외합니다. (기본 ON, UI에서 끌 수 있음)
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from src.models.schemas import SentenceRecord

# 기관/양식에서 자주 보는 문구·기호 (부분 일치)
_BOILERPLATE_SUBSTRINGS = (
    "기술 요약 정보",
    "매출 실적",
    "기관명",
    "한국전자기술연구원",
    "연구개발과제의 개요",
    "총괄책임자",
    "참여연구원",
    "비밀유지",
    "보안등급",
    "문서번호",
    "작성일자",
    "제출일",
    "양식",
    "서식",
    # 장표·서술 연결 문구 (내용 재사용이 아님)
    "장표에 있는 바와 같이",
    "다음과 같이",
    "아래와 같이",
    "바와 같이",
)

# 라벨/양식 줄로 보는 정규식
_BOILERPLATE_PATTERNS = [
    re.compile(r"[□■☐☑✓✗※★☆◆◇▶▷►]"),  # 양식 체크/불릿 기호
    re.compile(r"기관\s*명\s*[:：]"),
    re.compile(r"<\s*기관"),
    re.compile(r"^\s*\[[^\]]{1,40}\]\s*$"),  # [1차년도 개발내용] 같은 짧은 대괄호 제목
    re.compile(r"^\s*<[^>]{1,60}>\s*$"),  # < 기관명 : ... >
    re.compile(r"차년도\s*개발내용"),
    re.compile(r"연구개발\s*과제"),
    re.compile(r"누적\s*\)?\s*$"),
]

_YEAR_SECTION = re.compile(r"^[\[\(]?\s*\d\s*차\s*년도")

REASON_PATTERN = "양식 패턴"
REASON_LABEL = "라벨형"
REASON_COMMON = "파일공통"

# 파일공통 제외는 '짧은 양식 라벨'에만 적용.
# 긴 기술 문장이 여러 파일에 동일하게 있어도 유사 검토 대상이므로 남긴다.
DEFAULT_COMMON_MAX_LENGTH = 20


def common_file_threshold(n_files: int) -> int:
    """
    동일 문장이 이 개수 이상의 파일에 나타나면 '공통 양식'으로 본다.

    - 파일 2개: 둘 다(2)에 있을 때만
    - 파일 3개+: N-1개 이상 (한 파일에만 없는 공통 문구)
    """
    if n_files <= 1:
        return 2  # 사실상 비활성
    if n_files == 2:
        return 2
    return n_files - 1


def is_label_like(text: str) -> bool:
    """짧은 라벨·기호 위주 줄인지 휴리스틱으로 판별."""
    t = (text or "").strip()
    if not t:
        return True

    # 너무 짧은 제목/라벨
    if len(t) <= 20 and (
        t.startswith("[")
        or t.startswith("<")
        or t.startswith("□")
        or t.startswith("※")
        or _YEAR_SECTION.search(t)
    ):
        return True

    letters = re.findall(r"[가-힣a-zA-Z]", t)
    if not letters:
        return True

    # 공백 제외한 기호(□·:·<> 등) 비율 — 공백만으로는 라벨로 보지 않음
    compact = re.sub(r"\s+", "", t)
    symbols = re.findall(r"[^가-힣a-zA-Z0-9]", compact)
    if len(t) <= 40 and len(symbols) >= 2 and len(letters) <= 12:
        return True

    return False


def is_boilerplate_pattern(text: str) -> bool:
    """금칙어·양식 패턴에 해당하면 True."""
    t = (text or "").strip()
    if not t:
        return True
    lower = t.replace(" ", "")
    for s in _BOILERPLATE_SUBSTRINGS:
        if s.replace(" ", "") in lower or s in t:
            return True
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(t):
            return True
    return False


def is_short_common_candidate(text: str, max_length: int = DEFAULT_COMMON_MAX_LENGTH) -> bool:
    """
    파일공통 필터 대상인지 판별.

    여러 파일에 같아도, 짧은 항목명·단위·헤더만 제외하고
    본문급 기술 문장은 유사 문장 결과에 남긴다.
    """
    t = (text or "").strip()
    if not t:
        return True
    # 개조식 앞머리 제거 후 길이 판단
    body = re.sub(r"^[ㅇ\-–•·]\s*", "", t).strip()
    if len(body) <= max_length:
        return True
    # 길어도 라벨형이면 제외 후보
    if is_label_like(body):
        return True
    return False


def build_text_file_index(sentences: Iterable[SentenceRecord]) -> dict[str, set[str]]:
    """정규화 문장 → 등장 파일명 집합."""
    index: dict[str, set[str]] = defaultdict(set)
    for s in sentences:
        key = (s.normalized_text or "").strip()
        if key:
            index[key].add(s.file_name)
    return index


def _excluded_row(s: SentenceRecord, reason: str, *, file_count: int = 0) -> dict:
    norm = (s.normalized_text or s.text or "").strip()
    return {
        "file_name": s.file_name,
        "file_type": getattr(s, "file_type", "pdf") or "pdf",
        "location": s.location,
        "text": s.text,
        "normalized_text": norm,
        "sentence_id": s.sentence_id or "",
        "reason": reason,
        "file_count": file_count,
        "row_id": f"{s.file_name}||{s.location}||{norm}",
    }


def filter_boilerplate_sentences(
    sentences: list[SentenceRecord],
    *,
    n_files: int,
    use_patterns: bool = True,
    use_common_across_files: bool = True,
    common_max_length: int = DEFAULT_COMMON_MAX_LENGTH,
) -> tuple[list[SentenceRecord], dict, list[dict]]:
    """
    양식/공통 문장을 걸러낸다.

    파일공통은 common_max_length 이하의 짧은 문구(또는 라벨형)에만 적용한다.

    Returns:
        (남은 문장, 통계 dict, 제외 문장 목록)
    """
    stats = {
        "input": len(sentences),
        "removed_pattern": 0,
        "removed_label": 0,
        "removed_common": 0,
        "kept": 0,
        "common_threshold": common_file_threshold(n_files),
        "common_max_length": common_max_length,
        "n_files": n_files,
    }
    excluded: list[dict] = []
    if not sentences:
        stats["kept"] = 0
        stats["removed_total"] = 0
        return [], stats, excluded

    file_index = build_text_file_index(sentences)
    threshold = stats["common_threshold"]
    kept: list[SentenceRecord] = []

    for s in sentences:
        text = s.normalized_text or s.text or ""
        key = (s.normalized_text or "").strip()
        files = file_index.get(key, set())

        if use_patterns and is_boilerplate_pattern(text):
            stats["removed_pattern"] += 1
            excluded.append(_excluded_row(s, REASON_PATTERN, file_count=len(files)))
            continue
        if use_patterns and is_label_like(text):
            stats["removed_label"] += 1
            excluded.append(_excluded_row(s, REASON_LABEL, file_count=len(files)))
            continue
        if (
            use_common_across_files
            and len(files) >= threshold
            and is_short_common_candidate(text, max_length=common_max_length)
        ):
            stats["removed_common"] += 1
            excluded.append(_excluded_row(s, REASON_COMMON, file_count=len(files)))
            continue
        kept.append(s)

    stats["kept"] = len(kept)
    stats["removed_total"] = stats["input"] - stats["kept"]
    return kept, stats, excluded
