# Search RFP

Streamlit 기반 RFP 검색/질의응답 서비스입니다.

## 실행 방법

```bash
cd /Users/who/Desktop/code_it/project01/final_files/서비스_깃
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## 데이터 경로

기본값은 이 폴더의 부모 폴더를 데이터 기준 경로로 사용합니다.

```text
../data/chroma_export.json
../data/data_list_advanced.xlsx
../data/files_advanced/
../chroma_seol_qwen3/
```

다른 위치의 데이터를 사용하려면 `RFP_BASE_PATH`를 지정합니다.

```bash
RFP_BASE_PATH=/path/to/final_files streamlit run app.py
```

## LLM 실행 조건

로컬 Ollama 서버가 필요하며, 기본 모델은 `gemma4:e4b`입니다.
