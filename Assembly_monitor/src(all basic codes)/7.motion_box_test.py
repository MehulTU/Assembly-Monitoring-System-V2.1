import cv2

camera = cv2.VideoCapture(0) # Change the index to 1 or 2 if you have multiple cameras

x = 200
y = 100
width = 400
height = 300

previous_gray = None

while True:

    success, frame = camera.read()

    if not success:
        break

    roi = frame[y:y+height, x:x+width]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(gray, (5, 5), 0) # Apply Gaussian blur to reduce noise and improve motion detection

    if previous_gray is not None:

        difference = cv2.absdiff(
            gray,
            previous_gray
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3)
        ) # Compute the absolute difference between the current and previous grayscale images

        _, binary_mask = cv2.threshold(
            difference,
            10,
            255,
            cv2.THRESH_BINARY
        ) # Create a binary mask from the difference image

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        ) # Find contours in the binary mask

        for contour in contours:

            area = cv2.contourArea(contour) # Calculate the area of the contour

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

        cv2.imshow(
            "Binary Motion Mask",
            binary_mask
        )

    previous_gray = gray

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (0,255,0),
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