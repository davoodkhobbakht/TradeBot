import subprocess
import os
import signal
import platform
import sys

class BotManager:
    def __init__(self):
        self.process = None

        self.base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

        self.entry_file = os.path.join(self.base_dir, "main.py")

    def start(self, args=None):
        if self.process and self.process.poll() is None:
            return {"status": "already_running"}

        if not os.path.exists(self.base_dir):
            return {"status": "error", "message": f"Bot folder not found: {self.base_dir}"}

        cmd = [sys.executable, self.entry_file]  # Use sys.executable for reliability

        if args:
            cmd.extend(args)  
        else:
            cmd.append("--live")  # Default to live mode

        print("🚀 Running:", cmd)
        print("📂 CWD:", self.base_dir)

        self.process = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr→stdout for log streaming
            text=True,
            bufsize=1  
        )

        return {"status": "started", "pid": self.process.pid, "cmd": cmd}

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