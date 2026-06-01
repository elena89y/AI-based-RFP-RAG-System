# 🏛️ AI 기반 공공조달 RFP 분석 RAG 시스템 (AI-based-RFP-RAG-System)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-0.1+-1C3C3C?style=flat-square&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-Vector_DB-009ACD?style=flat-square"/>
  <img src="https://img.shields.io/badge/RAGAS-Evaluation-FF6B35?style=flat-square"/>
</p>

<p align="center">
  <b>AI8기 중급 프로젝트 | Team 1</b><br/>
  AI-based RFP RAG System for public procurement document analysis <br><br>
  유연정(PM) · 정설아(데이터처리) · 정봄(데이터처리) · 박찬영(Retrieval) · 박민성(Generation)
</p>

<p align="center">
  <a href="https://www.notion.so/AI8-_-_1-RFP-X-35e1fffab02c80d9ae52daa9a6b5fa5a"> 📋 협업노트/회의록 & 프로젝트 가이드</a>
  [팀 보고서 PDF]()
  
</p>

---

## 📌 프로젝트 개요

공공기관 발주 **RFP(제안요청서) 문서**는 방대한 분량과 복잡한 행정 언어로 구성되어 있어, 담당자가 핵심 조건을 빠르게 파악하기 어렵다는 실무적 문제가 있습니다.

본 프로젝트는 **RAG(Retrieval-Augmented Generation) 파이프라인**을 구축하여 공공조달 RFP 문서에 대한 정확한 질의응답 시스템을 개발하고, 이를 체계적인 **평가 데이터셋**으로 검증합니다.

| 항목 | 내용 |
|------|------|
| 데이터 출처 | 제공된 데이터셋 |
| 평가 데이터셋 | `eval_qa_combined_500.xlsx` (Q001–Q500, 5개 유형) |
| 평가 범위 | eval_batch 01–25 |
| 핵심 과제 | RAG 파이프라인 구축 + RAGAS 기반 성능 평가 |

---

## 🗂️ 파일 구조

```
AI-based-RFP-RAG-System/
│
├── data/                        # 원본 RFP 문서 및 전처리 데이터
│
├── final_data_01.ipynb          # 데이터 수집 및 전처리 파이프라인
├── prom09.ipynb                 # 프롬프트 엔지니어링 및 QA 생성
├── com_02.ipynb                 # 평가 데이터셋 통합 및 검수
├── retrieval_V9.ipynb           # RAG 검색 파이프라인 (최종 v9)
│
├── .gitignore
└── README.md
```

---

## 🔧 기술 스택

### 핵심 파이프라인

| 구성 요소 | 기술 |
|----------|------|
| **문서 로딩 / 파싱** | PyMuPDF (fitz), pdfplumber |
| **텍스트 청킹** | LangChain `RecursiveCharacterTextSplitter` |
| **임베딩 모델** | OpenAI `text-embedding-ada-002` |
| **벡터 DB** | FAISS |
| **검색 방식** | Similarity Search + MMR (Maximal Marginal Relevance) |
| **생성 모델** | OpenAI GPT-4o |
| **RAG 프레임워크** | LangChain `RetrievalQA` / `ConversationalRetrievalChain` |

### 평가 프레임워크

| 구성 요소 | 기술 |
|----------|------|
| **평가 프레임워크** | RAGAS |
| **평가 지표** | Faithfulness · Answer Relevancy · Context Precision · Context Recall |
| **QA 데이터셋** | GPT-4o 기반 자동 생성 + 수동 검수 |
| **데이터 포맷** | `.xlsx` (Q001–Q500, 5개 질문 유형) |

### 질문 유형 분류 (Type A–E)

| 유형 | 설명 |
|------|------|
| **Type A** | 단순 사실 확인 (수치, 날짜, 기관명 등) |
| **Type B** | 조건 및 자격 요건 |
| **Type C** | 절차 및 프로세스 |
| **Type D** | 비교 / 복합 조건 |
| **Type E** | 추론 및 종합 판단 |

---

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
git clone https://github.com/elena89y/AI-based-RFP-RAG-System.git
cd AI-based-RFP-RAG-System
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
cp .env.example .env
```

`.env` 파일에 아래 내용을 입력합니다:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 실행 순서

```
1. final_data_01.ipynb   → 문서 로딩 및 청킹, 벡터 DB 구축
2. prom09.ipynb          → 프롬프트 설계 및 QA 쌍 자동 생성
3. com_02.ipynb          → 평가셋 통합 (eval_qa_combined_500.xlsx)
4. retrieval_V9.ipynb    → RAG 검색 파이프라인 실행 및 RAGAS 평가
```

---

## 📊 RAG 파이프라인 구조

```
[RFP PDF 문서]
      │
      ▼
[문서 파싱 & 청킹]  ←  RecursiveCharacterTextSplitter
      │
      ▼
[임베딩 생성]  ←  text-embedding-ada-002
      │
      ▼
[FAISS 벡터 DB 저장]
      │
      ▼
[사용자 질의]
      │
      ▼
[유사도 검색 (Similarity / MMR)]
      │
      ▼
[GPT-4o 답변 생성]
      │
      ▼
[RAGAS 평가]  ←  Faithfulness / Relevancy / Precision / Recall
```

---

## 📁 평가 데이터셋

- **파일명**: `pm_data.xlsx`
- **규모**: 500개 QA 쌍 (Q001–Q500)
- **생성 방식**: GPT-4o 자동 생성 → 팀 수동 검수
- **범위**: eval_batch 01–25 (강사 가이드라인 기준)
- **구성**: question / answer / context / source_doc / question_type

---

## 👥 팀 구성 및 역할

| 이름 | 역할 |
|------|------|
| 유연정 | PM · 평가 설계 총괄 · eval_batch 통합 검수 |
| 정설아 | 데이터 전처리 |
| 정봄 | 데이터 전처리 |
| 박찬영 | Retrieval · 벡터 DB 구축 · 검색 최적화 |
| 박민성 | Generation · RAG 파이프라인 개발 |


---

## 📎 참고 자료

- [📋 협업 노트 / 회의록 (Notion)](https://www.notion.so/AI8-_-_1-RFP-X-35e1fffab02c80d9ae52daa9a6b5fa5a)
- [LangChain 공식 문서](https://docs.langchain.com)
- [RAGAS 공식 문서](https://docs.ragas.io)
- [중급프로젝트 가이드라인](https://codeit.notion.site/AI-1ee6fd228e8d80d4834bee9cef8f44c1)

---

<p align="center">
  <sub>AI8기 중급 프로젝트 | 2026 · Team 1</sub>
</p>
