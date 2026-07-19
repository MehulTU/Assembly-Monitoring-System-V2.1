import cv2

camera = cv2.VideoCapture(0)

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

    roi = frame[y:y+height, x:x+width]

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

        difference = cv2.absdiff(
            gray,
            previous_gray
        )

        _, binary_mask = cv2.threshold(
            difference,
            10,
            255,
            cv2.THRESH_BINARY
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )

        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # =====================================
        # Motion Score
        # =====================================

        moving_pixels = cv2.countNonZero(
            binary_mask
        )

        total_pixels = width * height

        motion_percentage = (
            moving_pixels / total_pixels
        ) * 100

        # =====================================
        # Contours
        # =====================================

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

                perimeter = cv2.arcLength(
                    contour,
                    True
                )

                M = cv2.moments(
                    contour
                )

                if M["m00"] != 0:

                    center_x = int(
                        M["m10"] / M["m00"]
                    )

                    center_y = int(
                        M["m01"] / M["m00"]
                    )

                    cv2.circle(
                        roi,
                        (center_x, center_y),
                        4,
                        (0, 255, 0),
                        -1
                    )

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

                cv2.putText(
                    roi,
                    f"A:{int(area)}",
                    (cx, cy - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )

                cv2.putText(
                    roi,
                    f"P:{int(perimeter)}",
                    (cx, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )

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

    previous_gray = gray.copy()

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

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()