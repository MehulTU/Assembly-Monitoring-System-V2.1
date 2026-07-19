import cv2
print("Program started")
print("VERSION 2 TEST")
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) #Open camera number 0
print("Camera opened:", camera.isOpened())

x = 200
y = 100
width = 400
height = 300

previous_gray = None

while True:
    success, frame = camera.read() #Take a picture and save it to "frame"
    print("Frame read success:", success)

    if not success:
        print ("failed to capture image")
        break

    roi = frame[y:y+height, x:x+width] #Extract the ROI from the frame
    print("ROI extracted")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #Convert the ROI to grayscale
    print("ROI converted to grayscale")

    if previous_gray is not None:
        difference = cv2.absdiff(gray, previous_gray)
        motion_value = difference.sum()

        print("Motion value:", motion_value)

        cv2.putText(
            frame,
            f"Motion Value: {motion_value}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2,
        )

        if motion_value > 500000:
            cv2.putText(
                frame,
                "Motion Detected!",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2,
            )
    else:
        motion_value = 0

    previous_gray = gray

    cv2.rectangle(
        frame,
        (x,y),
        (x+width,y+height),
        (0,255,0),
        2,
    )

    cv2.imshow("Motion Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print ("motion test complete")