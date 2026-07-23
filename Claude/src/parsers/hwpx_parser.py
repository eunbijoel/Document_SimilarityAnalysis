"""HWPX 파서 스텁. 현재 버전은 PDF만 지원합니다.

향후 구현 시 HWPX는 zip 컨테이너 + XML(lxml) 구조이므로 lxml로 content.xml을
파싱하는 방식을 권장합니다.
"""
from src.models.schemas import ParseResult


def parse_hwpx(file_name: str, file_bytes: bytes, **kwargs) -> ParseResult:
    return ParseResult(
        file_name=file_name,
        success=False,
        error_message="HWPX 지원은 다음 버전에서 추가될 예정입니다 (현재 버전은 PDF만 지원).",
    )
