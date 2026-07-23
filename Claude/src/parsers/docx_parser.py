"""DOCX 파서 스텁.

현재 버전(v1)의 필수 구현 범위는 PDF만 지원합니다.
이 파일은 향후 Word 문서 지원을 추가할 때 pdf_parser.py와 동일한 인터페이스
(parse_docx(file_name, file_bytes, ...) -> ParseResult)로 구현하기 위한 자리표시자입니다.
"""
from src.models.schemas import ParseResult


def parse_docx(file_name: str, file_bytes: bytes, **kwargs) -> ParseResult:
    return ParseResult(
        file_name=file_name,
        success=False,
        error_message="DOCX 지원은 다음 버전에서 추가될 예정입니다 (현재 버전은 PDF만 지원).",
    )
