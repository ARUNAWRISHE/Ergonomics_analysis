# utils/labeling.py
import os
import pandas as pd

def append_session_to_datasets(session_csv: str, label: str, pose_data_csv: str, pose_data_labeled_csv: str) -> int:
    if not os.path.exists(session_csv) or os.path.getsize(session_csv) == 0:
        return 0

    df = pd.read_csv(session_csv)

    os.makedirs(os.path.dirname(pose_data_csv), exist_ok=True)
    if not os.path.exists(pose_data_csv) or os.path.getsize(pose_data_csv) == 0:
        df.to_csv(pose_data_csv, index=False)
    else:
        df.to_csv(pose_data_csv, mode="a", header=False, index=False)

    dfl = df.copy()
    dfl["label"] = str(label)
    os.makedirs(os.path.dirname(pose_data_labeled_csv), exist_ok=True)
    if not os.path.exists(pose_data_labeled_csv) or os.path.getsize(pose_data_labeled_csv) == 0:
        dfl.to_csv(pose_data_labeled_csv, index=False)
    else:
        dfl.to_csv(pose_data_labeled_csv, mode="a", header=False, index=False)

    return len(df)
