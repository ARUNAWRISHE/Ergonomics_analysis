# login.py
import tkinter as tk
from tkinter import messagebox
import csv
import os
import subprocess
import sys

CREDENTIAL_FILE = "data/credential.csv"

def load_credentials():
    creds = {}
    if not os.path.exists(CREDENTIAL_FILE):
        raise FileNotFoundError(f"Credential file not found: {CREDENTIAL_FILE}")
    with open(CREDENTIAL_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row.get("username", "").strip()
            p = row.get("password", "").strip()
            if u:
                creds[u] = p
    return creds

class LoginApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Login")
        self.root.geometry("320x180")
        self.creds = load_credentials()
        self._build_ui()

    def _build_ui(self):
        tk.Label(self.root, text="Username:").pack(pady=(20,5))
        self.username_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.username_var).pack()

        tk.Label(self.root, text="Password:").pack(pady=(10,5))
        self.password_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.password_var, show="*").pack()

        tk.Button(self.root, text="Login", command=self._do_login).pack(pady=15)

    def _do_login(self):
        u = self.username_var.get().strip()
        p = self.password_var.get().strip()

        if u in self.creds and self.creds[u] == p:
            self.root.destroy()
            if u.lower() == "admin":
                # Launch admin.py with a --skip-login flag
                subprocess.Popen([sys.executable, "admin.py", "--skip-login"])
            else:
                subprocess.Popen([sys.executable, "live_detection_alarm.py"])
        else:
            messagebox.showerror("Login failed", "Invalid username or password")

def main():
    root = tk.Tk()
    app = LoginApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
