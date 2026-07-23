"""PPTX 파서 스텁. docx_parser.py 참고. 현재 버전은 PDF만 지원합니다."""
from src.models.schemas import ParseResult


def parse_pptx(file_name: str, file_bytes: bytes, **kwargs) -> ParseResult:
    return ParseResult(
        file_name=file_name,
        success=False,
        error_message="PPTX 지원은 다음 버전에서 추가될 예정입니다 (현재 버전은 PDF만 지원).",
    )
