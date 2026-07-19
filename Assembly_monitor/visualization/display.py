"""
visualization/display.py
================================
SUBSYSTEM: Visualization (SS9) — File 5 System Architecture

Draws the live display overlay on each frame — state, metrics, ROI,
bounding boxes, warmup indicator, and motion score bar.

This subsystem ONLY displays. It never changes any data.
That is the correct separation of concerns.

V2.1 confidence upgrade:
    - Detection boxes are COLOR-CODED by confidence:
        red    = low confidence   (< CONF_MEDIUM_THRESHOLD)
        yellow = medium confidence
        green  = high confidence  (>= CONF_HIGH_THRESHOLD)
    - Labels are drawn on a filled background so they are readable
      on any scene.
    - A confidence bar is drawn on top of every detection box.
    - A YOLO CONFIDENCE PANEL (top-right) shows, for every class:
        best confidence this frame, rolling average confidence,
        and detection count — so you can see model quality live.
"""

import cv2

# ---------- Colours ----------
COLOR_ACTIVE = (0, 165, 255)
COLOR_IDLE = (180, 180, 180)
COLOR_WARMUP = (50, 200, 255)

COLOR_ROI = (0, 200, 80)

COLOR_MOTION = (255, 100, 0)

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# Confidence colour coding (BGR)
COLOR_CONF_LOW = (0, 0, 255)       # red
COLOR_CONF_MEDIUM = (0, 215, 255)  # yellow
COLOR_CONF_HIGH = (0, 220, 0)      # green

# Confidence thresholds for the colour coding
CONF_MEDIUM_THRESHOLD = 0.40
CONF_HIGH_THRESHOLD = 0.60


def confidence_color(confidence):
    """Map a confidence value to a display colour."""
    if confidence >= CONF_HIGH_THRESHOLD:
        return COLOR_CONF_HIGH
    if confidence >= CONF_MEDIUM_THRESHOLD:
        return COLOR_CONF_MEDIUM
    return COLOR_CONF_LOW


class LiveDisplay:

    def __init__(self, roi_points=None):
        self._roi_points = roi_points

    def render(
        self,
        frame,
        motion_result,
        object_result,
        event_state,
        event_duration,
        metrics_summary,
        is_warmup,
        warmup_remaining,
    ):

        display = frame.copy()

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        if self._roi_points is not None:
            cv2.polylines(
                display,
                [self._roi_points],
                True,
                COLOR_ROI,
                2,
            )

        # ----------------------------------------------------
        # Motion boxes
        # ----------------------------------------------------

        for (x, y, w, h) in motion_result.bounding_boxes:

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                COLOR_MOTION,
                2,
            )

            cv2.putText(
                display,
                "Motion",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_MOTION,
                1,
            )

        # ----------------------------------------------------
        # YOLO detections (confidence-coded)
        # ----------------------------------------------------

        if object_result is not None:

            for obj in object_result.objects:
                self._draw_detection(display, obj)

            self._draw_confidence_panel(
                display,
                object_result,
                frame.shape,
            )

        # ----------------------------------------------------
        # Status Panel
        # ----------------------------------------------------

        if is_warmup:

            state_color = COLOR_WARMUP
            state_text = f"WARMUP ({warmup_remaining})"

        else:

            state_color = (
                COLOR_ACTIVE
                if "ACTIVE" in event_state
                else COLOR_IDLE
            )

            state_text = event_state

        self._draw_status_panel(
            display,
            state_text,
            state_color,
            event_duration,
            motion_result.motion_score,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        self._draw_metrics(
            display,
            metrics_summary,
            frame.shape,
        )

        return display

    # ==========================================================
    # Single YOLO detection (box + confidence bar + label)
    # ==========================================================

    def _draw_detection(self, frame, obj):

        x1, y1, x2, y2 = obj.bounding_box
        color = confidence_color(obj.confidence)

        # Bounding box (thicker when confidence is high)
        thickness = 3 if obj.confidence >= CONF_HIGH_THRESHOLD else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Confidence bar along the top edge of the box
        bar_width = max(1, int((x2 - x1) * min(obj.confidence, 1.0)))
        bar_y1 = max(0, y1 - 6)
        cv2.rectangle(
            frame,
            (x1, bar_y1),
            (x2, y1),
            (60, 60, 60),
            -1,
        )
        cv2.rectangle(
            frame,
            (x1, bar_y1),
            (x1 + bar_width, y1),
            color,
            -1,
        )

        # Label with filled background: "marker 43%"
        label = f"{obj.class_name} {obj.confidence * 100:.0f}%"

        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )

        label_y = max(text_h + 10, bar_y1 - 4)

        cv2.rectangle(
            frame,
            (x1, label_y - text_h - 6),
            (x1 + text_w + 8, label_y + baseline - 2),
            COLOR_BLACK,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    # ==========================================================
    # YOLO Confidence Panel (top-right)
    # ==========================================================

    def _draw_confidence_panel(self, frame, object_result, frame_shape):
        """
        Show per-class confidence statistics:

            DETECTIONS
            marker         43%  avg 38%  x2
            power_adapter  61%  avg 55%  x1

        'avg' is the rolling average of the best confidence over
        the last N frames (from ObjectDetector), which is much more
        stable to read than a per-frame number.
        """

        stats = getattr(object_result, "class_stats", None)

        frame_w = frame_shape[1]

        panel_w = 300
        row_h = 22
        rows = max(1, len(stats) if stats else 1)
        panel_h = 30 + rows * row_h

        x0 = frame_w - panel_w - 8
        y0 = 8

        cv2.rectangle(
            frame,
            (x0, y0),
            (x0 + panel_w, y0 + panel_h),
            COLOR_BLACK,
            -1,
        )
        cv2.rectangle(
            frame,
            (x0, y0),
            (x0 + panel_w, y0 + panel_h),
            (80, 80, 80),
            1,
        )

        cv2.putText(
            frame,
            "DETECTIONS",
            (x0 + 8, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_WHITE,
            1,
        )

        if not stats:
            cv2.putText(
                frame,
                "none",
                (x0 + 8, y0 + 20 + row_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_IDLE,
                1,
            )
            return

        y = y0 + 20 + row_h

        for class_name in sorted(stats.keys()):

            s = stats[class_name]
            color = confidence_color(s.best_confidence)

            # Mini confidence bar
            bar_x = x0 + 8
            bar_w = 60
            cv2.rectangle(
                frame,
                (bar_x, y - 10),
                (bar_x + bar_w, y - 2),
                (60, 60, 60),
                -1,
            )
            cv2.rectangle(
                frame,
                (bar_x, y - 10),
                (bar_x + int(bar_w * min(s.best_confidence, 1.0)), y - 2),
                color,
                -1,
            )

            text = (
                f"{class_name[:13]:<13} "
                f"{s.best_confidence * 100:3.0f}% "
                f"avg {s.rolling_confidence * 100:3.0f}% "
                f"x{s.count}"
            )

            cv2.putText(
                frame,
                text,
                (bar_x + bar_w + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
            )

            y += row_h

    # ==========================================================
    # Status Panel
    # ==========================================================

    def _draw_status_panel(
        self,
        frame,
        state_text,
        color,
        duration,
        score,
    ):

        cv2.rectangle(
            frame,
            (8, 8),
            (330, 95),
            COLOR_BLACK,
            -1,
        )

        cv2.rectangle(
            frame,
            (8, 8),
            (330, 95),
            color,
            2,
        )

        cv2.putText(
            frame,
            state_text,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        cv2.putText(
            frame,
            f"Duration : {duration:.2f}s",
            (16, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_WHITE,
            1,
        )

        cv2.putText(
            frame,
            f"Motion : {score:.2f} %",
            (16, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_WHITE,
            1,
        )

    # ==========================================================
    # Metrics
    # ==========================================================

    def _draw_metrics(
        self,
        frame,
        metrics,
        frame_shape,
    ):

        h = frame_shape[0]

        y = h - 80

        cv2.rectangle(
            frame,
            (8, y),
            (340, h - 8),
            COLOR_BLACK,
            -1,
        )

        cv2.rectangle(
            frame,
            (8, y),
            (340, h - 8),
            (80, 80, 80),
            1,
        )

        lines = [

            f"Active Events : {metrics.get('total_active_events',0)}",

            f"Idle Events   : {metrics.get('total_idle_events',0)}",

            f"Avg Step Time : {metrics.get('avg_step_time_sec',0):.2f}s",

            f"Cycle Time    : {metrics.get('avg_cycle_time_sec',0):.2f}s",

        ]

        for i, line in enumerate(lines):

            cv2.putText(
                frame,
                line,
                (16, y + 18 + i * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_WHITE,
                1,
            )
