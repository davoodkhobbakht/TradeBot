import subprocess
import os
import signal
import platform

class BotManager:
    def __init__(self):
        self.process = None

        # 🔥 FIX: always resolve project root safely
        self.base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        # ⚠️ your actual bot folder (relative to project root)
        self.bot_dir = os.path.join(self.base_dir, "")

        self.entry_file = "main.py"

    def start(self, args=None):
        if self.process and self.process.poll() is None:
            return {"status": "already_running"}

        if not os.path.exists(self.bot_dir):
            return {
                "status": "error",
                "message": f"Bot folder not found: {self.bot_dir}"
            }

        cmd = ["python", self.entry_file]

        if args:
            cmd += args
        else:
            cmd += ["--live"]

        print("🚀 Running:", cmd)
        print("📂 CWD:", self.bot_dir)

        self.process = subprocess.Popen(
            cmd,
            cwd=self.bot_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        return {
            "status": "started",
            "pid": self.process.pid,
            "cwd": self.bot_dir,
            "cmd": cmd
        }

    def stop(self):
        if not self.process:
            return {"status": "not_running"}

        if platform.system() == "Windows":
            self.process.terminate()
        else:
            os.kill(self.process.pid, signal.SIGTERM)

        self.process = None
        return {"status": "stopped"}

    def status(self):
        if self.process and self.process.poll() is None:
            return {"status": "running", "pid": self.process.pid}

        return {"status": "stopped"}