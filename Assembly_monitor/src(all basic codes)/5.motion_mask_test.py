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
        print("Failed")
        break

    roi = frame[y:y+height, x:x+width]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    if previous_gray is not None:

        difference = cv2.absdiff(
            gray,
            previous_gray
        )

        cv2.imshow(
            "Motion Mask",
            difference
        )

    previous_gray = gray

    cv2.rectangle(
        frame,
        (x,y),
        (x+width,y+height),
        (0,255,0),
        2
    )

    cv2.imshow(
        "Original Frame",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()