# Generation 파트 실험 보고서

**프로젝트명**: RFP 문서 기반 RAG Q&A 시스템  
**가상 회사**: 입찰메이트 (B2G 입찰지원 컨설팅 스타트업)  
**담당**: Generation (민성)  
**기간**: 2026-05-12 ~ 2026-06-04

---

## 1. 모델 선정 실험

### 1.1 시나리오 B — OpenAI API

프롬프트 품질 개선 실험의 기준을 확보하기 위해 OpenAI API 모델을 고정 사용하였다.

| 항목 | 내용 |
|------|------|
| LLM | gpt-5.4-mini |
| 임베딩 | text-embedding-3-small (1536차원) |
| 목적 | 프롬프트 실험 기준 모델로 고정, 변인 통제 |

### 1.2 시나리오 A — 로컬 LLM 비교 실험

API 의존도 및 비용 문제를 해결하기 위해 Ollama 기반 로컬 모델을 검토하였다. Mac Mini M4 24GB 환경에서 실행 가능한 모델을 대상으로 한국어 성능, 응답 속도, Tool Calling 지원 여부, 수학 계산 정확도를 기준으로 비교하였다.

#### LLM 비교 실험 결과

| 모델 | 파라미터 | 메모리 | 한국어 | 수학 | Tool Calling | 평가 결과 |
|------|----------|--------|--------|------|-------------|----------|
| qwen2.5:7b | 7B | ~5GB | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | 수학 연산 약점으로 탈락 |
| qwen3:14b | 14B | ~10GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | 속도 이슈로 보조 사용 |
| phi-4 14B | 14B | ~10GB | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ | 한국어 약점으로 탈락 |
| EXAONE-3.5 7.8B | 7.8B | ~6GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | **프롬프트 실험 채택** |
| EXAONE 4.0 | 1.2B/32B | - | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ | Ollama 미공식 지원으로 탈락 |
| **gemma4:e4b** | 4B(MoE) | ~10GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | **최종 서비스 모델 선정** |

#### 최종 모델 선정 이유

**프롬프트 실험용 — EXAONE-3.5 7.8B**
- LG AI Research 개발, 한국어 특화 모델
- 8개 후보 모델 중 faithfulness 1.0, answer_relevancy 0.55로 1위
- Tool Calling 미지원이지만 프롬프트 품질 측정 목적으로 채택

**서비스용 — gemma4:e4b**
- Google 개발, 140개 언어 지원 (한국어 포함)
- 한국어 + 수학 계산 + Tool Calling 세 가지 동시 충족
- Mac M4에서 Tool Calling 실검증 완료 (57 t/s)
- EXAONE이 Tool Calling 미지원으로 최종 교체 결정

### 1.3 임베딩 모델 비교

| 항목 | 시나리오 B | 시나리오 A |
|------|-----------|-----------|
| 모델 | text-embedding-3-small | Qwen/Qwen3-Embedding-0.6B |
| 차원 | 1536 | 1024 |
| 특징 | OpenAI 제공 | 다국어 119개 언어, MTEB 최상위 |
| 비용 | API 비용 발생 | $0 (로컬) |

---

## 2. 프롬프트 개선 실험 (prom_01 ~ prom_09)

### 2.1 실험 개요

| 항목 | 내용 |
|------|------|
| 실험 파일 | prom_01.ipynb ~ prom_09.ipynb |
| 고정 평가셋 | 18개 (Type A 6개, Type B 8개, Filter 2개, None 2개) |
| 평가 지표 | faithfulness, answer_relevancy, context_precision (RAGAS) |
| 비교 구조 | V1 (Re-writing 없음) vs V2 (Re-writing 적용) |
| 최종 버전 | SYSTEM_PROMPT v7e |

### 2.2 버전별 핵심 변경 이력

| 버전 | 핵심 변경 내용 | 주요 효과 |
|------|-------------|----------|
| v1 | 역할 정의 + Guardrail + 출처 명시 | Baseline |
| v2 | CoT 추가 + **답변/근거/출처** 3섹션 구조화 + Self-Correction | 답변 구조 안정화, 출처 명시율 향상 |
| v3 | 시스템 프롬프트 금액 인식 규칙 추가 | 예산 추출 정확도 향상 |
| v4 | 다중 기관 비교 질문 처리 규칙 추가 | 비교 질문 답변 품질 향상 |
| v5 | 기간 질문 추정 금지 규칙 추가 | 기간 오답률 감소 |
| v6 | 금액 미기재/비공개 처리 규칙 추가 | 금액 오생성 방지 |
| **v7e** | Deterministic Answer + TYPO_REPLACEMENTS + Evidence Card | 금액/기간 정확도 대폭 향상 |

### 2.3 시나리오 B 프롬프트 실험 결과 (V1 기준)

| 실험 | faithfulness | answer_relevancy | context_precision |
|------|-------------|-----------------|------------------|
| Dense Only v1 | 0.787 | 0.312 | 0.569 |
| Dense Only v2 | 0.795 | 0.336 | 0.581 |
| BM25 Hybrid v1 | 0.913 | 0.365 | 0.491 |
| BM25 Hybrid v2 | 0.916 | 0.531 | 0.641 |
| Hybrid+Reranker v1 | 0.863 | 0.369 | **0.722** |

> Type B answer_relevancy: V1 0.28 → V2 0.53 (+89%), Query Re-writing 효과 확인

### 2.4 v7e 핵심 기능 상세

#### Deterministic Answer 로직

LLM 호출 없이 규칙 기반으로 직접 답변을 생성하여 정확도와 속도를 동시에 개선하였다.

| 유형 | 발동 조건 | 처리 방식 |
|------|----------|----------|
| 단순 예산 | 금액 키워드 + 단일 사업 | 청크에서 핵심 금액 라인 추출 |
| 금액 + 비율 | %, 공제, 차감 키워드 | 순차 차감 계산 |
| 다중 금액 합산/차액 | 금액 2개 이상 질문 내 포함 | 질문 제시 수치로 직접 계산 |
| 기간 | 개월, 기간 키워드 | 개월 수 직접 추출 |

#### TYPO_REPLACEMENTS

오타·비문체 표현 37개를 사전 정의하여 자동 보정한다.

```
"굥희머학교" → "경희대학교"
"시쓰탬"     → "시스템"
"에싼"       → "예산"
"구쭉"       → "구축"  등
```

#### Evidence Card

질문 관련 핵심 수치·금액 라인을 별도 카드로 구성하여 LLM 컨텍스트에 우선 제공한다. 수치 정확도 향상에 기여하였다.

### 2.5 Query Re-writing (ConversationManagerV2)

모호한 지시어 감지 시 독립 질문으로 재작성하여 검색 정확도를 높였다.

**발동 조건**
```
히스토리 존재 AND (모호한 지시어 포함 OR 오타 감지)
모호한 지시어: "그것", "해당", "앞서", "방금", "거기", "그쪽" 등
```

| 지표 | V1 (재작성 없음) | V2 (재작성 적용) | 개선 |
|------|----------------|----------------|------|
| Type B answer_relevancy | 0.28 | **0.53** | +89% |

---

## 3. 검색 파이프라인 고도화 (시나리오 B)

### 3.1 단계별 실험 결과 (전체)

| 실험 | faithfulness | answer_relevancy | context_precision |
|------|-------------|-----------------|------------------|
| Dense Only (p05) | 0.787 | 0.312 | 0.569 |
| BM25 Hybrid (p06) | **0.913** | 0.365 | 0.491 |
| **Hybrid+Reranker (p07)** | 0.863 | **0.369** | **0.722** |

### 3.2 Type B (후속 질문) 비교

| 실험 | faithfulness | answer_relevancy | context_precision |
|------|-------------|-----------------|------------------|
| Dense Only | 0.896 | 0.281 | 0.755 |
| BM25 Hybrid | 0.916 | 0.531 | 0.641 |
| **Hybrid+Reranker** | **0.965** | 0.460 | **0.842** |

### 3.3 핵심 인사이트

- **Dense → Hybrid**: BM25 키워드 보완으로 faithfulness·answer_relevancy ↑, 단 BM25 노이즈로 context_precision 일시 ↓
- **Hybrid → Reranker**: BGE-Reranker가 노이즈 제거 → context_precision 0.49 → 0.72 (+47%)
- **Type B answer_relevancy**: Query Re-writing 효과로 0.28 → 0.53 (+89%)

---

## 4. 시나리오 A 최종 평가 (gemma4:e4b)

### 4.1 평가 환경

| 항목 | 내용 |
|------|------|
| LLM | gemma4:e4b (Ollama, Mac M4) |
| 임베딩 | Qwen3-Embedding-0.6B |
| 벡터DB | ChromaDB (rfp_seol_chunks_qwen3) |
| 평가셋 | pm_data.xlsx 500건 (Stratified 20건 샘플) |
| Tool | calculate, format_currency, date_diff, vat_calculator, extract_entities, related_docs_finder |

### 4.2 RAGAS 점수

| 구분 | faithfulness | answer_relevancy | context_precision |
|------|-------------|-----------------|------------------|
| V1 전체 | 미측정* | 0.633 | 0.214 |
| **V2 전체** | 미측정* | **0.614** | **0.228** |
| V1 TypeC | 미측정* | 0.691 | 0.563 |
| V2 TypeC | 미측정* | 0.671 | 0.556 |

> \* gemma4:e4b가 RAGAS 내부 JSON 형식 미준수로 faithfulness 측정 불가

### 4.3 커스텀 지표 (20건 샘플)

| 지표 | V1 | V2 |
|------|----|----|
| Hit@3 | 0.800 | 0.800 |
| Hit@5 | 0.800 | 0.800 |
| Hit@10 | **0.850** | 0.850 |
| MRR@5 | **0.800** | 0.725 |
| nDCG@5 | **0.800** | 0.751 |
| Coverage@5 | 0.750 | 0.750 |
| 출처 누락률 | **0.000** | **0.000** |
| 과잉 거부율 | 0.222 | **0.167** |
| 금액 오답률 | **0.286** | 0.375 |
| 평균 응답시간 | 20.82s | 20.09s |

### 4.4 수정 이력 및 개선 효과

| 수정 내용 | 이전 | 이후 | 개선 |
|----------|------|------|------|
| evidence_docs 키 연결 | 0건 | 20건 | Retrieval 평가 가능 |
| ground_truth 키 수정 | 빈값 | 정상 | RAGAS 정상 실행 |
| 계산 로직 (금액+비율) | 10,800,000원 (오답) | **109,350,000원** (정답) | ✅ |
| 계산 로직 (금액 2개 합산) | 과잉 거부 | **1,665,000,000원** (정답) | ✅ |
| 과잉 거부율 | 0.389 | **0.222** | -43% |
| 금액 오답률 | 0.500 | **0.286** | -43% |

### 4.5 시나리오 A vs B 비교

| 항목 | 시나리오 B (p07) | 시나리오 A (gemma4) |
|------|----------------|-------------------|
| LLM | gpt-5.4-mini | gemma4:e4b |
| 임베딩 | text-embedding-3-small | Qwen3-Embedding-0.6B |
| context_precision | **0.722** | 0.214 |
| answer_relevancy | 0.369 | **0.633** |
| API 비용 | 유 | **$0** |

---

## 5. 부가 실험 및 연구

### 5.1 HWP 파싱 이슈 해결

Mac 환경에서 olefile 바이너리 파서가 HWP 제어코드를 한자/특수문자로 오독하여 garbage 문자를 대량 생성하는 문제가 발생하였다.

```
olefile 파싱 결과 예시:
  '\x02捤獥\x00\x00\x00\x00\x02\x02汤捯...'  ← 95% garbage
  평균 73,824자 추출 → 실제 사용 가능 3,600자 (4.9%)
```

**해결**: hwp5txt (CLI 기반 HWP 파서) primary + 바이너리 파서 표 추출 병행 방식으로 전환하여 파싱 품질을 정상화하였다.

### 5.2 ChromaDB 플랫폼 이식

팀원이 Windows에서 구축한 ChromaDB를 Mac M4로 이식하는 과정에서 HNSW 바이너리 비호환 오류가 발생하였다.

```
InternalError: Error loading hnsw index
→ HNSW 바이너리가 Windows(x86_64) 전용 빌드
→ Mac M4(ARM) 환경에서 로드 불가
```

**해결**: Windows에서 임베딩 벡터 추출(embeddings.npy) → Mac에서 재삽입하여 HNSW 재생성

| 방법 | 소요 시간 |
|------|----------|
| Mac에서 재임베딩 | 30~60분 |
| **벡터 이식 방식** | **3~5분** |

### 5.3 Tool Calling 도입

gemma4:e4b의 Tool Calling 기능을 활용하여 계산 정확도를 향상시켰다. EXAONE-3.5에서 gemma4:e4b로 모델을 교체한 핵심 이유이기도 하다.

| 도구 | 기능 | 실제 적용 사례 |
|------|------|-------------|
| calculate | 수식 계산 | 예산 차액/합계 계산 |
| format_currency | 원↔억원↔만원 변환 | 금액 단위 표준화 |
| date_diff | 날짜 간격 계산 | 착수일~납기 기간 계산 |
| vat_calculator | VAT 포함/제외 계산 | 계약금액 환산 |
| extract_entities | RFP 핵심 엔티티 추출 | 사업 정보 구조화 |
| related_docs_finder | 유사 사업 검색 | ChromaDB 유사도 검색 |

### 5.4 평가셋 품질 분석 (pm_data.xlsx 500건)

| 문제 | 건수 | 비율 | 영향 |
|------|------|------|------|
| 누수 질문 (파일명 직접 포함) | 139건 | 27.8% | Hit@k 과대평가 |
| 참조 문서 편중 | 41개 반복 | - | 우연 적중률 상승 |

누수 질문으로 인해 실제 검색 성능이 과대 평가될 수 있으므로 clean 셋(361건) 기준 재평가가 필요하다.

---

## 6. 알려진 이슈 및 개선 방향

| 이슈 | 원인 | 개선 방향 |
|------|------|----------|
| faithfulness 측정 불가 | gemma4가 RAGAS JSON 형식 미준수 | RAGAS 전용 API 모델(gpt-5.4-mini) 사용 |
| 과잉 거부율 22% | 예산 청크가 top-7 검색 범위 외 | ANSWER_TOP_K 증가, 검색 전략 개선 |
| E타입 오타 보정 실패 | 심한 오타 TYPO 사전 미등록 | TYPO_REPLACEMENTS 사전 확장 |
| Tool 자율 호출률 낮음 | SYSTEM_PROMPT에 지시 추가 후 개선 | TOOL_INSTRUCTION 지속 개선 |
