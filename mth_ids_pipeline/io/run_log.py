"""Log de execução: espelha mensagens no terminal e em arquivo."""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from mth_ids_pipeline.io.subprocess_env import configure_stdio_utf8, utf8_subprocess_env


class RunLog:
    """Escreve progresso no console e em ``*.log`` (append, com timestamp)."""

    def __init__(self, log_path: Path) -> None:
        configure_stdio_utf8()
        self.path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = log_path.open("a", encoding="utf-8")
        self.emit(f"=== sessão iniciada — log: {log_path} ===")

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> RunLog:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.emit(f"=== sessão encerrada com erro: {exc_type.__name__}: {exc} ===")
        else:
            self.emit("=== sessão concluída ===")
        self.close()

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def emit(self, message: str) -> None:
        """Mensagem de progresso (timestamp só no arquivo)."""
        print(message, flush=True)
        self._file.write(f"[{self._timestamp()}] {message}\n")
        self._file.flush()

    def emit_raw(self, line: str) -> None:
        """Saída de subprocesso (mesmo texto no terminal e no log)."""
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def run_subprocess(self, cmd: list[str], *, cwd: Path) -> float:
        """Executa comando capturando stdout/stderr linha a linha no log."""
        self.emit(">> " + " ".join(cmd))
        t0 = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=utf8_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            self.emit_raw(line.rstrip("\n"))
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
        return time.time() - t0
