import unittest
import time
import os
import csv
import tempfile

from monitor.gpu_monitor import GPUMonitor


class TestGPUMonitor(unittest.TestCase):
    def test_start_stop(self):
        monitor = GPUMonitor(interval=0.1)
        monitor.start()
        time.sleep(0.3)
        monitor.stop()

        results = monitor.get_results()
        self.assertGreater(len(results), 0)

    def test_elapsed_seconds_increasing(self):
        monitor = GPUMonitor(interval=0.1)
        monitor.start()
        time.sleep(0.35)
        monitor.stop()

        results = monitor.get_results()
        self.assertGreater(len(results), 2)

        for i in range(1, len(results)):
            self.assertGreaterEqual(
                results[i]["elapsed_seconds"], results[i - 1]["elapsed_seconds"]
            )

    def test_csv_export(self):
        monitor = GPUMonitor(interval=0.1)
        monitor.start()
        time.sleep(0.3)
        monitor.stop()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            temp_path = f.name

        try:
            monitor.export_csv(temp_path)

            self.assertTrue(os.path.exists(temp_path))

            with open(temp_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertGreater(len(rows), 0)
            self.assertIn("elapsed_seconds", rows[0])
            self.assertIn("power_watts", rows[0])
            self.assertIn("gpu_clock_mhz", rows[0])
            self.assertIn("mem_clock_mhz", rows[0])
        finally:
            os.unlink(temp_path)

    def test_clear(self):
        monitor = GPUMonitor(interval=0.1)
        monitor.start()
        time.sleep(0.2)
        monitor.stop()

        self.assertGreater(len(monitor.get_results()), 0)

        monitor.clear()
        self.assertEqual(len(monitor.get_results()), 0)

    def test_context_manager(self):
        with GPUMonitor(interval=0.1) as monitor:
            time.sleep(0.3)

        results = monitor.get_results()
        self.assertGreater(len(results), 0)

    def test_multiple_start_stop(self):
        monitor = GPUMonitor(interval=0.1)

        monitor.start()
        time.sleep(0.2)
        monitor.stop()

        results1 = monitor.get_results()
        count1 = len(results1)

        monitor.start()
        time.sleep(0.2)
        monitor.stop()

        results2 = monitor.get_results()
        count2 = len(results2)

        self.assertGreater(count2, count1)


if __name__ == "__main__":
    unittest.main()
