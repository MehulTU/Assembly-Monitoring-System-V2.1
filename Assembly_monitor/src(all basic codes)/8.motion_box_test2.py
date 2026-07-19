import cv2

# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)  # Use the appropriate camera index (0, 1, etc.)

# ==========================================
# ROI SETTINGS
# ==========================================

x = 200
y = 100
width = 400
height = 300

# ==========================================
# ASSEMBLY ZONE
# ==========================================

assembly_x = 220
assembly_y = 120
assembly_w = 100
assembly_h = 100

# ==========================================
# VARIABLES
# ==========================================

previous_gray = None
motion_history = []

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = camera.read()

    if not success:
        print("Failed to capture frame")
        break

    # =====================================
    # EXTRACT ROI
    # =====================================

    roi = frame[y:y+height, x:x+width]

    # Draw assembly zone
    cv2.rectangle(
        roi,
        (assembly_x, assembly_y),
        (assembly_x + assembly_w,
         assembly_y + assembly_h),
        (255, 0, 0),
        2
    )

    cv2.putText(
        roi,
        "ASSEMBLY ZONE",
        (assembly_x, assembly_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1
    )

    # =====================================
    # GRAYSCALE
    # =====================================

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # =====================================
    # BLUR
    # =====================================

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # =====================================
    # MOTION DETECTION
    # =====================================

    if previous_gray is not None:

        difference = cv2.absdiff(
            gray,
            previous_gray
        )

        # =================================
        # THRESHOLD
        # =================================

        _, binary_mask = cv2.threshold(
            difference,
            20,
            255,
            cv2.THRESH_BINARY
        )

        # =================================
        # DILATION
        # =================================

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3)
        )

        binary_mask = cv2.dilate(
            binary_mask,
            kernel,
            iterations=2
        )

        # =================================
        # CONTOURS
        # =================================

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            # Smaller threshold for screws
            if area > 50:

                cx, cy, cw, ch = cv2.boundingRect(
                    contour
                )

                # =========================
                # BOUNDING BOX
                # =========================

                cv2.rectangle(
                    roi,
                    (cx, cy),
                    (cx + cw, cy + ch),
                    (0, 0, 255),
                    2
                )

                # =========================
                # CENTER POINT
                # =========================

                center_x = cx + cw // 2
                center_y = cy + ch // 2

                cv2.circle(
                    roi,
                    (center_x, center_y),
                    4,
                    (0, 255, 0),
                    -1
                )

                # =========================
                # SHOW AREA
                # =========================

                cv2.putText(
                    roi,
                    f"A:{int(area)}",
                    (cx, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )

                # =========================
                # TRACE HISTORY
                # =========================

                motion_history.append(
                    (center_x, center_y)
                )

                if len(motion_history) > 50:
                    motion_history.pop(0)

                # =========================
                # ASSEMBLY ZONE CHECK
                # =========================

                if (
                    assembly_x <= center_x <= assembly_x + assembly_w
                    and
                    assembly_y <= center_y <= assembly_y + assembly_h
                ):

                    cv2.putText(
                        roi,
                        "OBJECT ENTERED ASSEMBLY ZONE",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

        # =================================
        # DRAW MOTION TRACE
        # =================================

        for i in range(1, len(motion_history)):

            cv2.line(
                roi,
                motion_history[i - 1],
                motion_history[i],
                (0, 255, 255),
                2
            )

        # =================================
        # SHOW MASK
        # =================================

        cv2.imshow(
            "Binary Motion Mask",
            binary_mask
        )

    # =====================================
    # UPDATE PREVIOUS FRAME
    # =====================================

    previous_gray = gray.copy()

    # =====================================
    # DRAW ROI BOX
    # =====================================

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

    # =====================================
    # SHOW MAIN WINDOW
    # =====================================

    cv2.imshow(
        "Motion Box Detection",
        frame
    )

    # =====================================
    # KEYBOARD
    # =====================================

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================

camera.release()
cv2.destroyAllWindows()