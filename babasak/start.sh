#!/bin/sh
uvicorn backend.main:app --host 0.0.0.0 --port 9000 &
streamlit run app.py --server.port $DATABRICKS_APP_PORT --server.address 0.0.0.0
