import cv2
print ("starting camera...")
camera = cv2.VideoCapture(1) #Open camera number 2
print ("camera opened successfully")
while True:
    success, frame = camera.read() #Take a picture and save it to "frame"

    if not success:
        print ("failed to capture image")
        break
    cv2.imshow("Camera Test", frame) #show picture in window 

    key = cv2.waitKey(1) #check keyboard every second

    if key == ord('q'): #if "q" is pressed, quit
        print ("quitting camera test")
        break