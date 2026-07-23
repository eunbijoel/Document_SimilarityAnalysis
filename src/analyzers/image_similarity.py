"""이미지 유사도 분석 모듈 (perceptual hash 기반).

CLIP 등 의미 기반 이미지 분석은 이번 버전 범위가 아닙니다. 이후 버전에서
`compute_semantic_similarity()` 같은 함수를 이 모듈에 추가하는 방식으로 확장할 수 있도록
phash 계산과 비교 로직을 분리해 두었습니다.
"""
import imagehash

from src.models.schemas import ImageRecord
from src.parsers.image_parser import bytes_to_image
from src.utils.config import image_verdict


def compute_phash(image_record: ImageRecord) -> ImageRecord:
    """이미지의 phash를 계산해 image_record.phash에 채워 반환합니다.
    이미지를 열 수 없으면 phash는 None으로 유지됩니다."""
    pil_image = bytes_to_image(image_record.image_bytes)
    if pil_image is None:
        return image_record
    try:
        image_record.phash = str(imagehash.phash(pil_image))
    except Exception:  # noqa: BLE001
        image_record.phash = None
    return image_record


def _pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def find_similar_image_pairs(
    images: list[ImageRecord],
    distance_threshold: int = 8,
    include_same_file: bool = False,
) -> list[dict]:
    """pHash 해밍 거리 기준으로 유사 이미지 쌍을 찾습니다.
    이미지 개수는 문장 수보다 훨씬 적은 경우가 많아 단순 이중 루프로 처리합니다."""
    valid = [img for img in images if img.phash]
    n = len(valid)
    results = []
    seen: set[tuple[int, int]] = set()

    for i in range(n):
        hash_i = imagehash.hex_to_hash(valid[i].phash)
        for j in range(i + 1, n):
            if not include_same_file and valid[i].file_name == valid[j].file_name:
                continue
            key = _pair_key(i, j)
            if key in seen:
                continue
            hash_j = imagehash.hex_to_hash(valid[j].phash)
            distance = hash_i - hash_j  # 해밍 거리
            if distance > distance_threshold:
                continue
            seen.add(key)
            results.append(
                {
                    "file_a": valid[i].file_name,
                    "location_a": valid[i].location,
                    "image_id_a": valid[i].image_id,
                    "image_bytes_a": valid[i].image_bytes,
                    "file_b": valid[j].file_name,
                    "location_b": valid[j].location,
                    "image_id_b": valid[j].image_id,
                    "image_bytes_b": valid[j].image_bytes,
                    "phash_distance": int(distance),
                    "verdict": image_verdict(int(distance)),
                }
            )
    return results
