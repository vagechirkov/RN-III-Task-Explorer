#!/bin/bash
set -e -x

ls -la

streamlit run app/streamlit.py --server.port=5000 --server.address=0.0.0.0
