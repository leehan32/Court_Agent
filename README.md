# ⚖️ B2B Legal AI Assistant (LangGraph 기반)

본 프로젝트는 한국 로펌을 위한 **온프레미스 B2B 법률 AI 어시스턴트**를 구축하기 위한 청사진입니다. LangGraph로 제어되는 에이전트형 워크플로우를 통해 변호사의 문서 분석, 근거 탐색, 초안 작성, 반론 시뮬레이션까지 하나의 도구 안에서 처리하며, 로컬 인프라와 RLS 기반 보안으로 민감한 사건 데이터를 보호합니다.

## 🎯 최우선 목표

* **완전한 비공개 환경**: 로컬 LLM과 온프레미스 배포 옵션으로 의뢰인 정보가 외부로 유출되지 않도록 설계합니다.
    (현제 프로젝트에서는ocr의 타 ocr사용중 한글깨짐 이슈에 네이버 api를 임시적으로 쓰고 있습니다. 차 후 paddleocr같은 오픈소스형으로 바꾸려합니다.)
* **신뢰 가능한 법률 근거**: 공용 법령·판례 DB와 변호사 전용 DB를 결합한 RAG로, 모든 답변이 검증 가능한 출처를 갖도록 합니다.
* **변호사 개인화**: 사용자가 수정한 문장을 피드백 DB에 누적해, 변호사별 문체와 논리 스타일을 학습합니다.
* **워크플로우 통합**: 파일 업로드부터 HWP/DOCX 서면 초안 다운로드까지, 실제 업무 흐름을 그대로 지원합니다.

## 🧭 시스템 블루프린트

> 아래 구조는 구현을 위한 "최종 청사진"으로, 각 파트는 모듈화된 파이프라인으로 구성됩니다.

### 1. 데이터 수집 파이프라인 (File Ingestion)

| 구성 요소 | 역할 |
| --- | --- |
| **파일 파서** | `pyhwp`, `python-docx`, `PyMuPDF`로 HWP/DOCX/PDF에서 텍스트와 이미지를 추출 |
| **OCR 엔진** | 네이버 CLOVA OCR API로 스캔 이미지·비텍스트 PDF를 정확한 한국어 텍스트로 변환 |
| **텍스트 분할기** | LangChain `RecursiveCharacterTextSplitter`로 법률 문맥을 유지한 채 청크 단위로 분할 |

### 2. 데이터베이스 아키텍처 (PostgreSQL + Redis)

* **공용 법령 DB (`statutes_archive`)**: 조문별 버전·시행일을 포함한 로컬 사본으로, 외부 API 장애 시에도 신뢰도 유지.
* **공용 판례 DB (`case_archive`)**: `batch_learn.py`로 임베딩한 판례 벡터를 저장해 RAG 2순위 검색에 활용.
* **전용 사건 DB (RLS 적용)**: `user_id`/`firm_id` 컬럼과 Row-Level Security 정책으로, 로펌별·변호사별 데이터 격리를 보장.
* **피드백 DB**: 변호사가 수정한 `revised_text`를 `user_id`와 함께 기록해 개인화 학습에 활용.

### 3. AI 코어 (LangGraph 상태 머신)

LangGraph 그래프는 다음 노드로 구성됩니다:

1. `summarize_node`: 업로드 문서를 요약하고 핵심 쟁점을 추출.
2. `rag_chain_node`: `user_id` 기준 전용 DB → 공용 DB 순으로 근거를 검색(RAG).
3. `drafting_node`: 수집된 근거로 법률 서면 초안을 작성하고 HWP/DOCX 템플릿에 매핑.
4. `simulation_node`: 예상 반론과 불리한 판례를 생성해 변호사에게 대비 전략 제공.

> **LLM 선택**: OpenAI API 대신 Llama 3, SOLAR, Mixtral 등 로컬/온프레미스 LLM을 기본값으로 사용합니다.

### 4. 애플리케이션 & 배포 전략

* **HWP/DOCX Export**: 초안 결과를 즉시 서면 템플릿에 반영해 다운로드 가능.
* **피드백 UI**: 변호사가 초안을 직접 수정하고 "내 DB에 저장" 버튼으로 피드백을 축적.
* **배포 모델**
  * **On-Premise**: 전 구성 요소(DB, LLM, 애플리케이션)를 로펌 내부 서버에 설치.
  * **Private Cloud**: RLS로 격리된 SaaS 환경을 제공하되, 데이터는 한국 리전 내에서만 저장.

### 5. 차별화 포인트 (vs. ChatGPT)

1. **데이터 기밀성**: 외부 LLM에 의존하지 않고 온프레미스 또는 폐쇄형 클라우드에서 동작.
2. **법률 특화 근거**: 실제 판례·법령 사본을 바탕으로 한 RAG로 환각을 차단하고 모든 답변에 근거를 부여.
3. **지속적 개인화**: 변호사별 피드백 DB를 통해 사용할수록 사용자 맞춤형 초안이 생성.
4. **업무 자동화**: 업로드 → 분석 → 초안 작성 → HWP/DOCX 다운로드까지 한 번의 워크플로우로 통합.



##차후 개발 청사진
![Animation](https://github.com/user-attachments/assets/f28cf24d-d500-4bf0-8857-3b3d361d9304)


## ⚙️ 개발 환경 설정

```bash
# 레포지토리를 클론합니다.
git clone <repository-url>
cd Court_Agent

# 환경 변수 파일을 준비합니다.
cp .env.example .env

# LLM 및 OCR 자격 정보를 설정합니다.
#   LLM_PROVIDER=local       # 로컬/온프레미스 LLM 경로
#   CLOVA_OCR_SECRET=...     # 네이버 CLOVA OCR API 키
#   DATABASE_URL=...         # PostgreSQL 접속 정보

# 의존성 설치
pip install -r requirements.txt

# 데이터베이스 및 캐시 서버 실행
docker-compose up -d
```

## 🚀 실행 워크플로우

1. **사전 학습 (선택)**: 공개 판례 데이터로 벡터 DB를 초기화합니다.
   ```bash
   python batch_learn.py
   ```
2. **단일 사건 시뮬레이션**: 전체 파이프라인을 한 번 실행합니다.
   ```bash
   python main.py
   ```
3. **벤치마크 측정**: 학습 전/후 성능을 비교하여 품질을 확인합니다.
   ```bash
   python benchmark.py --mode untrained   # 초기 상태 평가
   python benchmark.py --mode trained     # 피드백 반영 후 평가
   ```

## 🗄️ 데이터 관리

| 작업 | 명령 |
| --- | --- |
| 상태 확인 | `docker ps` |
| DB/캐시 데이터 조회 | PostgreSQL(`localhost:5433`), Redis(`localhost:6379`)에 툴(DBeaver 등)로 접속 |
| 전체 초기화 | `docker-compose down -v` |

---

이 README는 현재 시스템 목표와 구조를 문서화한 것으로, 구현 과정에서 발생하는 변경 사항은 본 문서를 기준으로 업데이트하세요.
