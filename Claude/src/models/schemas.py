"""데이터 구조 정의 모듈.

문장 레코드와 이미지 레코드의 스키마를 dataclass로 정의합니다.
다른 모듈들은 이 스키마를 기준으로 데이터를 주고받습니다.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PageRecord:
    """PDF 페이지 전체 텍스트 (페이지 유사도 비교용)."""

    file_name: str
    page_number: int
    text: str  # 정제된 페이지 전체 텍스트
    location: str = ""

    def __post_init__(self) -> None:
        if not self.location:
            self.location = f"페이지 {self.page_number}"


@dataclass
class SentenceRecord:
    """추출된 문장 하나에 대한 메타데이터."""

    file_name: str
    file_type: str
    location: str  # 예: "페이지 3", "슬라이드 5", "문단 12"
    text: str  # 원문 (정제 전)
    normalized_text: str  # 공백/줄바꿈 정리된 텍스트 (비교용)
    sentence_id: str = ""  # 고유 ID (파일명 + 인덱스)


@dataclass
class ImageRecord:
    """추출된 이미지 하나에 대한 메타데이터."""

    file_name: str
    location: str
    image_id: str
    width: int
    height: int
    image_bytes: bytes = field(repr=False, default=b"")
    phash: Optional[str] = None  # perceptual hash (hex string), 유사도 분석 단계에서 채워짐


@dataclass
class ParseResult:
    """파일 하나를 파싱한 결과."""

    file_name: str
    success: bool
    sentences: list = field(default_factory=list)
    images: list = field(default_factory=list)
    pages: list = field(default_factory=list)  # PageRecord 목록
    error_message: str = ""
