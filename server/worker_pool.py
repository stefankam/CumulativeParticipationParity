"""Thread-safe active/standby physical worker management."""

import threading


class PhysicalWorkerPoolExhausted(RuntimeError):
    """Raised when a failed active worker has no healthy standby replacement."""


class PhysicalWorkerPool:
    def __init__(self):
        self._lock = threading.Lock()
        self.active = []
        self.standby = []
        self.quarantined = set()

    def configure(self, physical_ids, active_count):
        physical_ids = list(dict.fromkeys(physical_ids))
        active_count = int(active_count)
        if active_count < 1:
            raise ValueError("PHYSICAL_CONTAINER_LIMIT must be positive")
        if len(physical_ids) < active_count:
            raise PhysicalWorkerPoolExhausted(
                f"Need {active_count} active physical workers, but only "
                f"{len(physical_ids)} healthy workers registered."
            )
        with self._lock:
            self.active = physical_ids[:active_count]
            self.standby = physical_ids[active_count:]
            self.quarantined = set()

    def active_snapshot(self):
        with self._lock:
            return list(self.active)

    def quarantine_and_promote(self, failed_worker):
        with self._lock:
            if failed_worker not in self.active:
                raise PhysicalWorkerPoolExhausted(
                    f"Physical worker {failed_worker} failed outside the active pool."
                )
            failed_index = self.active.index(failed_worker)
            self.quarantined.add(failed_worker)
            if not self.standby:
                raise PhysicalWorkerPoolExhausted(
                    f"Physical worker {failed_worker} failed and no standby "
                    "workers remain; aborting the experiment."
                )
            replacement = self.standby.pop(0)
            self.active[failed_index] = replacement
            return replacement
