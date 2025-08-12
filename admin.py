# admin.py
import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import os
import shutil
import sys

from utils.io_paths import Paths
from utils.capture_modal import run_modal_capture_session
from utils.labeling import append_session_to_datasets
from utils.training import train_and_save_model

# Default admin credentials (only used when not launched from login.py)
USERNAME = "admin"
PASSWORD = "12345"

class AdminApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ergonomics Admin")
        self.root.geometry("380x300")
        self.paths = Paths()
        self.status_var = tk.StringVar(value="")

    # ----------------- LOGIN UI -----------------
    def _login_ui(self):
        self._clear()
        tk.Label(self.root, text="Login", font=("Arial", 16)).pack(pady=10)
        frm = tk.Frame(self.root)
        frm.pack(pady=5)
        tk.Label(frm, text="Username").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        tk.Label(frm, text="Password").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()
        tk.Entry(frm, textvariable=self.username_var).grid(row=0, column=1, padx=5, pady=5)
        tk.Entry(frm, textvariable=self.password_var, show="*").grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Login", command=self._handle_login).pack(pady=10)

    # ----------------- ADMIN MAIN UI -----------------
    def _main_ui(self):
        self._clear()
        tk.Label(self.root, text="Admin Panel", font=("Arial", 16)).pack(pady=10)

        tk.Button(self.root, text="Good Pose Training", width=26,
                  command=lambda: self._capture_session_modal("good")).pack(pady=6)
        tk.Button(self.root, text="Bad Pose Training", width=26,
                  command=lambda: self._capture_session_modal("bad")).pack(pady=6)
        tk.Button(self.root, text="Train Model", width=26,
                  command=self._train_model).pack(pady=6)

        # Download log button
        tk.Button(self.root, text="Download Log (XLSX)", width=26,
                  command=self._download_log).pack(pady=6)

        tk.Label(self.root, textvariable=self.status_var, fg="gray").pack(pady=8)

    # ----------------- LOGIN HANDLER -----------------
    def _handle_login(self):
        u = self.username_var.get().strip()
        p = self.password_var.get().strip()
        if u == USERNAME and p == PASSWORD:
            self._main_ui()
        else:
            messagebox.showerror("Login failed", "Invalid credentials")

    # ----------------- CAPTURE -----------------
    def _capture_session_modal(self, label: str):
        try:
            self.status_var.set(f"Status: capturing {label} pose (press 'q' to stop)...")
            self.root.update_idletasks()
            os.makedirs(self.paths.sessions_dir, exist_ok=True)
            os.makedirs(self.paths.data_dir, exist_ok=True)

            csv_path = run_modal_capture_session(self.paths.sessions_dir, label=label)
            if csv_path and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                rows = append_session_to_datasets(csv_path, label,
                                                  self.paths.pose_data_csv,
                                                  self.paths.pose_data_labeled_csv)
                self.status_var.set(f"Status: captured {rows} rows for '{label}' and appended to datasets.")
            else:
                self.status_var.set("Status: capture canceled or no frames recorded.")
        except Exception as e:
            self.status_var.set("Status: error during capture")
            messagebox.showerror("Capture error", str(e))

    # ----------------- TRAINING -----------------
    def _train_model(self):
        def _train():
            try:
                self.status_var.set("Status: training model...")
                os.makedirs(self.paths.models_dir, exist_ok=True)
                report = train_and_save_model(self.paths.pose_data_labeled_csv, self.paths.model_path)
                self.status_var.set("Status: training complete.")
                messagebox.showinfo("Training complete", report)
            except Exception as e:
                self.status_var.set("Status: training failed.")
                messagebox.showerror("Training error", str(e))
        threading.Thread(target=_train, daemon=True).start()

    # ----------------- DOWNLOAD LOG -----------------
    def _download_log(self):
        try:
            log_path = self.paths.bad_posture_xlsx
            if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                messagebox.showwarning("No log found", "Log file not found. Run live detection to generate it.")
                return

            default_name = os.path.basename(log_path)
            save_to = filedialog.asksaveasfilename(
                title="Save bad posture log",
                initialfile=default_name,
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")]
            )
            if not save_to:
                return
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            shutil.copyfile(log_path, save_to)
            messagebox.showinfo("Download complete", f"Saved log to:\n{save_to}")
        except Exception as e:
            messagebox.showerror("Download failed", str(e))

    def _clear(self):
        for w in self.root.winfo_children():
            w.destroy()


# ----------------- ENTRY POINT -----------------
def main():
    root = tk.Tk()
    app = AdminApp(root)
    # If launched with --skip-login, go directly to main panel
    if "--skip-login" in sys.argv:
        app._main_ui()
    else:
        app._login_ui()
    root.mainloop()

if __name__ == "__main__":
    main()
