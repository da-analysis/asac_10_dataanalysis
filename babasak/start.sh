#!/bin/sh
# uvicorn 에러를 파일로 저장 — chatbot 뷰에서 Connection refused 시 읽어서 표시
uvicorn backend.main:app --host 0.0.0.0 --port 9000 --log-level debug > /tmp/uvicorn.log 2>&1 &
streamlit run app.py --server.port $DATABRICKS_APP_PORT --server.address 0.0.0.0
