import cv2

camera = cv2.VideoCapture(0)  # Change the index if needed

x = 200
y = 100
width = 400
height = 300

previous_gray = None

while True:

    success, frame = camera.read()

    if not success:
        break

    print(frame.shape)

    roi = frame[y:y + height, x:x + width]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    if previous_gray is not None:

        # ---------------------------------
        # Frame Difference
        # ---------------------------------

        difference = cv2.absdiff(
            gray,
            previous_gray
        )

        # ---------------------------------
        # Threshold
        # ---------------------------------

        _, binary_mask = cv2.threshold(
            difference,
            10,
            255,
            cv2.THRESH_BINARY
        )

        # ---------------------------------
        # Morphological Filtering
        # ---------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )

        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # ==========================================
        # Motion Score
        # ==========================================

        moving_pixels = cv2.countNonZero(
            binary_mask
        )

        total_pixels = width * height

        motion_percentage = (
            moving_pixels / total_pixels
        ) * 100

        # ==========================================
        # Contours
        # ==========================================

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area > 50:

                cx, cy, cw, ch = cv2.boundingRect(
                    contour
                )

                cv2.rectangle(
                    roi,
                    (cx, cy),
                    (cx + cw, cy + ch),
                    (0, 0, 255),
                    2
                )

        # ==========================================
        # Display Motion Score
        # ==========================================

        cv2.putText(
            roi,
            f"Motion: {motion_percentage:.2f} %",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Binary Motion Mask",
            binary_mask
        )

    previous_gray = gray

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Motion Box Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()