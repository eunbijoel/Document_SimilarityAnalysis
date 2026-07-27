"""판정 기준 및 기본값을 한 곳에서 관리하는 설정 모듈.

여기 있는 값들을 바꾸면 앱 전체(문장/이미지 유사도 판정, 기본 UI 값)에 반영됩니다.
"""

# --- 문장 유사도 ---
SENTENCE_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # 다국어(한국어 포함) 지원 모델

DEFAULT_SENTENCE_THRESHOLD = 0.85  # 이 미만은 결과에 포함하지 않음 (짧은 구절 오탐 완화)
DEFAULT_MIN_SENTENCE_LENGTH = 12  # 너무 짧은 연결 문구 제외
DEFAULT_TOP_K = 5  # NearestNeighbors에서 문장 하나당 검색할 이웃 개수

# 유사도 -> 판정 라벨 매핑 (내림차순으로 정렬되어 있어야 함)
SENTENCE_VERDICT_LEVELS = [
    (1.00, "동일 문장"),
    (0.90, "매우 유사"),
    (0.85, "유사 가능성"),
]


def sentence_verdict(score: float) -> str:
    for threshold, label in SENTENCE_VERDICT_LEVELS:
        if score >= threshold:
            return label
    return "유사도 낮음"


# --- 페이지 유사도 (초안 compare_page_texts 기준) ---
DEFAULT_PAGE_THRESHOLD = 0.72  # 페이지 전체 텍스트 유사도 기준
PAGE_TEXT_EMBED_MAX_CHARS = 8000  # 임베딩 입력 상한
PAGE_TEXT_PREVIEW_CHARS = 500  # 결과/미리보기 표시 상한


# --- 이미지 유사도 ---
# 초안(pdf_similarity_checker) 기본값: 가로 180 / 세로 120 — 로고·아이콘 노이즈 완화
DEFAULT_MIN_IMAGE_WIDTH = 180
DEFAULT_MIN_IMAGE_HEIGHT = 120
# 하위 호환: 단일 값 UI가 필요할 때 가로 기준을 사용
DEFAULT_MIN_IMAGE_SIZE = DEFAULT_MIN_IMAGE_WIDTH
DEFAULT_PHASH_DISTANCE_THRESHOLD = 8  # 이 값 이하이면 유사 이미지로 판정 (0=완전 동일)

IMAGE_VERDICT_LEVELS = [
    (0, "동일 이미지"),
    (5, "매우 유사"),
    (10, "유사 가능성"),
]


def image_verdict(hamming_distance: int) -> str:
    for threshold, label in IMAGE_VERDICT_LEVELS:
        if hamming_distance <= threshold:
            return label
    return "유사도 낮음"


# --- 안전장치 ---
DEFAULT_MAX_RESULTS = 500  # 결과 표/다운로드에 포함할 최대 쌍 개수
DEFAULT_MAX_SENTENCES = 5000  # 대용량 PDF 대비 문장 비교 hard-cap (초안과 동일)
MAX_SENTENCES_WARNING = 5000  # hard-cap 도달 시 사용자에게 안내
EMBEDDING_BATCH_SIZE = 64
