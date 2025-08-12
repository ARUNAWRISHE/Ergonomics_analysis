# utils/io_paths.py
import os
from datetime import datetime

class Paths:
    def __init__(self):
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(utils_dir)

        self.data_dir = os.path.join(self.project_root, "data")
        self.sessions_dir = os.path.join(self.data_dir, "sessions")
        self.logs_dir = os.path.join(self.data_dir, "logs")
        self.models_dir = os.path.join(self.project_root, "models")
        self.assets_dir = os.path.join(self.project_root, "assets")

        self.pose_data_csv = os.path.join(self.data_dir, "pose_data.csv")
        self.pose_data_labeled_csv = os.path.join(self.data_dir, "pose_data_labeled.csv")
        self.bad_posture_xlsx = os.path.join(self.logs_dir, "bad_posture_log.xlsx")
        self.model_path = os.path.join(self.models_dir, "posture_model.pkl")
        self.beep_wav = os.path.join(self.assets_dir, "beep.wav")

    @staticmethod
    def timestamp_name(prefix: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        return f"{prefix}_{ts}.csv"
