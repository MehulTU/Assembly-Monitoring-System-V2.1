import cv2

# Open camera number 1
camera = cv2.VideoCapture(1)
if not camera.isOpened():
    raise RuntimeError("Could not open camera 1")

while True:
    # Take a picture and save it to "frame"
    success, frame = camera.read()
    if not success:
        print("failed to capture image")
        break

    # Region of interest (ROI) is the area of the image we want to analyze
    # We will use the top left corner of the image as our ROI
    x = 200
    y = 100
    width = 400
    height = 300

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),  # Green rectangle
        2,  # Thickness of the rectangle
    )
    roi = frame[y : y + height, x : x + width]  # Extract the ROI from the frame
    cv2.imshow("ROI Only", roi)  # show only the ROI in a separate window
    cv2.imshow("ROI Test", frame)  # show picture in window
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # if "q" is pressed, quit
        print("quitting ROI test")
        break

camera.release()
cv2.destroyAllWindows()
print("ROI test complete")