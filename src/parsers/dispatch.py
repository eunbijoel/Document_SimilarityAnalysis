"""확장자별 문서 파서 디스패치."""
from __future__ import annotations

from typing import Callable, Optional

from src.models.schemas import ParseResult
from src.parsers.docx_parser import parse_docx
from src.parsers.hwp_parser import parse_hwp
from src.parsers.hwpx_parser import parse_hwpx
from src.parsers.pdf_parser import parse_pdf
from src.parsers.pptx_parser import parse_pptx
from src.parsers.text_parser import parse_text

SUPPORTED_EXTENSIONS = (
    "pdf",
    "docx",
    "pptx",
    "hwp",
    "hwpx",
    "txt",
    "md",
    "text",
    "csv",
)

_PARSERS: dict[str, Callable[..., ParseResult]] = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "pptx": parse_pptx,
    "hwp": parse_hwp,
    "hwpx": parse_hwpx,
    "txt": parse_text,
    "md": parse_text,
    "text": parse_text,
    "csv": parse_text,
}


def get_extension(file_name: str) -> str:
    if "." not in file_name:
        return ""
    return file_name.rsplit(".", 1)[-1].lower().strip()


def parse_document(
    file_name: str,
    file_bytes: bytes,
    min_sentence_length: int = 8,
    min_image_size: Optional[int] = None,
    min_image_width: Optional[int] = None,
    min_image_height: Optional[int] = None,
) -> ParseResult:
    """파일 확장자에 맞는 파서로 문장·페이지·이미지를 추출합니다."""
    ext = get_extension(file_name)
    parser = _PARSERS.get(ext)
    if parser is None:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        return ParseResult(
            file_name=file_name,
            success=False,
            error_message=f"지원하지 않는 형식입니다 (.{ext or '?'}). 지원: {supported}",
        )
    return parser(
        file_name,
        file_bytes,
        min_sentence_length=min_sentence_length,
        min_image_size=min_image_size,
        min_image_width=min_image_width,
        min_image_height=min_image_height,
    )
