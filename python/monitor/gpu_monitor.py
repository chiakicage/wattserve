import threading
import time
import csv
from typing import List, Dict, Optional

import pynvml


class GPUMonitor:
    def __init__(self, gpu_index: int = 0, interval: float = 0.1):
        self.gpu_index = gpu_index
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._records: List[Dict] = []
        self._handle = None
        self._start_time: Optional[float] = None

    def _init_nvml(self):
        if self._handle is None:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

    def _collect(self):
        self._init_nvml()
        self._start_time = time.time()
        while self._running:
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                gpu_clock = pynvml.nvmlDeviceGetClock(
                    self._handle,
                    pynvml.NVML_CLOCK_GRAPHICS,
                    pynvml.NVML_CLOCK_ID_CURRENT,
                )
                mem_clock = pynvml.nvmlDeviceGetClock(
                    self._handle,
                    pynvml.NVML_CLOCK_MEM,
                    pynvml.NVML_CLOCK_ID_CURRENT,
                )

                elapsed = time.time() - self._start_time
                record = {
                    "elapsed_seconds": round(elapsed, 3),
                    "power_watts": round(power, 2),
                    "gpu_clock_mhz": gpu_clock,
                    "mem_clock_mhz": mem_clock,
                }

                with self._lock:
                    self._records.append(record)
            except Exception:
                pass

            time.sleep(self.interval)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_results(self) -> List[Dict]:
        with self._lock:
            return self._records.copy()

    def clear(self):
        with self._lock:
            self._records.clear()

    def export_csv(self, filepath: str):
        records = self.get_results()
        if not records:
            return

        fieldnames = [
            "elapsed_seconds",
            "power_watts",
            "gpu_clock_mhz",
            "mem_clock_mhz",
        ]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
