"""
analytics/metrics.py
================================
SUBSYSTEM: Analytics (SS6) — File 5 System Architecture

This is NOT image processing. This is time math.

Decision (SS5) tells us: "an ACTIVE event just ended, it lasted 6.2 seconds"
Analytics calculates:    "step time, cycle time, idle time, averages"

V1 METRICS (from File 8 Sheet 1 / File 5 SS6):
  Step completion time  → how long each assembly action took
  Cycle duration        → time from one ACTIVE event to the next
  Idle duration         → how long the system was idle between events

IMPORTANT:
  Analytics does NOT decide what happened.
  Analytics only measures time.
  That separation matches the architecture exactly.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional
from decision.event_detector import AssemblyEvent


@dataclass
class SessionMetrics:
    """
    Running summary of all metrics for the current recording session.
    Updated every time a new event is logged.
    """
    active_events:         List[AssemblyEvent] = field(default_factory=list)
    idle_events:           List[AssemblyEvent] = field(default_factory=list)

    # Step times (duration of each ACTIVE event)
    step_times:            List[float] = field(default_factory=list)

    # Idle times (duration of each IDLE event)
    idle_times:            List[float] = field(default_factory=list)

    # Cycle times (time from start of one ACTIVE to start of next ACTIVE)
    cycle_times:           List[float] = field(default_factory=list)

    _last_active_start:    Optional[float] = field(default=None, repr=False)

    @property
    def avg_step_time(self) -> float:
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0.0

    @property
    def avg_idle_time(self) -> float:
        return sum(self.idle_times) / len(self.idle_times) if self.idle_times else 0.0

    @property
    def avg_cycle_time(self) -> float:
        return sum(self.cycle_times) / len(self.cycle_times) if self.cycle_times else 0.0

    @property
    def total_active_events(self) -> int:
        return len(self.active_events)

    @property
    def total_idle_events(self) -> int:
        return len(self.idle_events)


class MetricsCalculator:
    """
    Receives assembly events and calculates running metrics.
    Maps to Analytics subsystem (SS6) in the system architecture.
    """

    def __init__(self):
        self.metrics = SessionMetrics()

    def process_event(self, event: AssemblyEvent):
        """
        Update metrics when a new event is received from Decision (SS5).

        Args:
            event: AssemblyEvent from EventDetector
        """
        if "ACTIVE" in event.event_type:
            self.metrics.active_events.append(event)
            self.metrics.step_times.append(event.duration_sec)

            # Calculate cycle time (time between start of consecutive ACTIVE events)
            if self.metrics._last_active_start is not None:
                cycle = event.start_time - self.metrics._last_active_start
                self.metrics.cycle_times.append(cycle)

            self.metrics._last_active_start = event.start_time

        elif event.event_type == "IDLE":
            self.metrics.idle_events.append(event)
            self.metrics.idle_times.append(event.duration_sec)

    def get_summary(self) -> dict:
        """Return a flat dictionary of current metrics for display/logging."""
        m = self.metrics
        return {
            "total_active_events": m.total_active_events,
            "total_idle_events":   m.total_idle_events,
            "avg_step_time_sec":   round(m.avg_step_time,  2),
            "avg_idle_time_sec":   round(m.avg_idle_time,  2),
            "avg_cycle_time_sec":  round(m.avg_cycle_time, 2),
            "last_step_time_sec":  round(m.step_times[-1], 2) if m.step_times else 0.0,
        }
