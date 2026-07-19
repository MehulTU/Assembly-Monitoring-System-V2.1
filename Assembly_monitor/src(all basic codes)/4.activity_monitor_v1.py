import cv2

camera = cv2.VideoCapture(0)

x = 200
y = 100
width = 400
height = 300

THRESHOLD = 500000

previous_gray = None

frame_counter = 0
previous_state = "IDLE"

while True:

    success, frame = camera.read()

    if not success:
        print("Failed to capture image")
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

        motion_score = difference.sum()

        if motion_score > THRESHOLD:
            state = "ACTIVE"
            color = (0,0,255)
        else:
            state = "IDLE"
            color = (0,255,0)

        frame_counter += 1

        if frame_counter % 30 == 0:
            print(
                f"Motion Score: {motion_score:.0f}"
            )

        if state != previous_state:
            print(
                f"State changed to: {state}"
            )
            previous_state = state

        cv2.rectangle(
            frame,
            (x,y),
            (x+width,y+height),
            color,
            2
        )

        cv2.putText(
            frame,
            f"State: {state}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Score: {int(motion_score)}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    previous_gray = gray

    cv2.imshow(
        "Assembly Monitoring",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()