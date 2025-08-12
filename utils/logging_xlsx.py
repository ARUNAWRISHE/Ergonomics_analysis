# utils/logging_xlsx.py
import os
import pandas as pd

def append_bad_event(xlsx_path: str, row: dict):
    os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)
    df_row = pd.DataFrame([row])
    if os.path.exists(xlsx_path) and os.path.getsize(xlsx_path) > 0:
        try:
            existing = pd.read_excel(xlsx_path)
            out = pd.concat([existing, df_row], ignore_index=True)
        except Exception:
            out = df_row
    else:
        out = df_row
    out.to_excel(xlsx_path, index=False)
