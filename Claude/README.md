# 문서 간 유사 콘텐츠 분석기

여러 PDF 문서를 업로드해 동일하거나 유사한 **문장**과 **이미지**를 찾아주는
내부 검토용 Streamlit 도구입니다.

> 이 도구는 표절 판정 시스템이 아닙니다. 여러 보고서·발표자료·계획서 사이에서
> 동일하거나 유사한 문장/그림이 재사용되었는지 검토하는 참고 자료를 제공할 뿐이며,
> 결과는 "동일" / "매우 유사" / "유사 가능성"으로만 표시됩니다.

## 1. 실행 방법

```bash
# 1) (최초 1회) 가상환경 생성 권장
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) 패키지 설치
pip install -r requirements.txt

# 3) 앱 실행
streamlit run app.py
```

브라우저가 자동으로 열리지 않으면 터미널에 표시되는 `http://localhost:8501` 주소로 접속하세요.

### 모델 다운로드 관련 안내

문장 유사도 분석에는 다국어(한국어 포함) sentence-transformers 모델
(`paraphrase-multilingual-MiniLM-L12-v2`)을 사용합니다.

- **최초 실행 시**: 인터넷에서 모델 파일(수백 MB)을 자동으로 내려받습니다. 인터넷 연결이 필요합니다.
- **이후 실행부터는** `~/.cache/torch/sentence_transformers` (또는 `~/.cache/huggingface`)에
  저장된 로컬 캐시를 사용하므로 인터넷 없이도 실행됩니다.
- 외부 API(OpenAI 등)는 사용하지 않으며, 모든 임베딩 계산은 로컬(CPU)에서 수행됩니다. GPU가 없어도 동작합니다.

## 2. 사용 방법

1. 사이드바에서 문장/이미지 판정 기준을 설정합니다 (기본값 그대로 사용해도 됩니다).
2. PDF 파일 2개 이상을 업로드합니다.
3. "분석 시작" 버튼을 누릅니다.
4. 탭에서 결과를 확인합니다: 분석 요약 / 유사 문장 / 유사 이미지 / 처리 로그
5. 하단에서 CSV(문장/이미지) 및 XLSX(통합 리포트)를 다운로드합니다.

## 3. 구현된 기능

- 여러 PDF 동시 업로드, 파일별 파싱 성공/실패 표시, 일부 실패해도 나머지는 계속 분석
- 페이지 단위 텍스트 추출 → **페이지 유사도** + 문장 분리(마침표·개조식 `ㅇ`/`-`/`•`) → 짧은/숫자만 문장 제외
- 완전히 동일한 문장은 문자열 비교로 우선 탐지 (임베딩 계산 없이)
- 다국어 sentence-transformers 임베딩 + `NearestNeighbors`(문장) / cosine 행렬(페이지)
- 문장 비교 **hard-cap** (기본 5000)으로 대용량 PDF 보호
- PDF 임베디드 이미지 추출 (기본 최소 가로 180 / 세로 120 — 로고·아이콘 제외)
- perceptual hash(pHash) 기반 이미지 유사도
- 분석 요약 / **유사 페이지** / 유사 문장(표+expander) / 유사 이미지 / 처리 로그 탭
- CSV(페이지·문장·이미지) + XLSX(Similar Pages 시트 포함) 다운로드

## 4. 현재 제한사항 (v1)

- **PDF만 지원**합니다. DOCX/PPTX/HWPX는 `src/parsers/`에 인터페이스만 맞춘 스텁 파일을
  만들어 두었고(`docx_parser.py`, `pptx_parser.py`, `hwpx_parser.py`), 실제 파싱 로직은
  구현되어 있지 않습니다 (다음 버전 확장 포인트).
- 이미지 유사도는 pHash(모양/색상 기반)만 사용하며, CLIP 등 의미 기반 이미지 유사도는
  포함하지 않습니다. `src/analyzers/image_similarity.py`에 향후 추가할 수 있도록 구조를 분리해
  두었습니다.
- 문장 분리는 정교한 자연어 문장 분리기가 아닌 규칙 기반(마침표/줄바꿈 등) 방식입니다.
- 대용량 PDF(수백 페이지, 수백 MB)는 파싱과 이미지 추출에 시간이 걸릴 수 있습니다. 진행률
  표시줄로 현재 처리 상황을 확인할 수 있습니다.

## 5. 프로젝트 구조

```text
app.py                      # Streamlit UI 및 전체 흐름 조정
requirements.txt
src/
├── parsers/
│   ├── pdf_parser.py        # PDF 텍스트+이미지 추출 (구현됨)
│   ├── docx_parser.py       # 스텁 (다음 버전)
│   ├── pptx_parser.py       # 스텁 (다음 버전)
│   ├── hwpx_parser.py       # 스텁 (다음 버전)
│   └── image_parser.py      # 이미지 바이트 -> PIL 변환 공용 헬퍼
├── analyzers/
│   ├── text_similarity.py   # 완전일치 탐지 + 임베딩/NearestNeighbors 유사도
│   └── image_similarity.py  # pHash 기반 이미지 유사도
├── models/
│   └── schemas.py           # SentenceRecord / ImageRecord / ParseResult
└── utils/
    ├── config.py            # 판정 기준 등 전역 설정값
    ├── text_processing.py   # 문장 분리/정제
    └── result_export.py     # CSV/XLSX 내보내기
```

## 6. 검증용 파일

단위 테스트 폴더는 제거했습니다. 우선 아래 실제 PDF로 앱을 돌려 확인합니다.

```text
../Test_files/
├── GITCC_연차보고서_KETI_수정_최종.pdf
└── 스위스_최종보고서_최종_날인_증빙.pdf
```

```bash
streamlit run app.py
```

업로드 시 `Test_files`의 PDF 2개를 선택해 분석하면 됩니다.
(스위스 보고서는 용량이 크므로 첫 분석에 시간이 걸릴 수 있습니다.)

## 7. 다음 버전에서 추가할 수 있는 기능

- DOCX/PPTX/HWPX 파서 실제 구현
- CLIP 등 의미 기반 이미지 유사도(그림 내용이 비슷하지만 스타일이 다른 경우 탐지)
- 문장 분리기를 더 정교한 한국어 문장 분리 로직으로 교체
- 유사 문장/이미지 결과에 대한 사용자 피드백(오탐 표시) 기능
- 대용량 파일 처리 속도 개선 (멀티프로세싱, 이미지 다운스케일 등)
