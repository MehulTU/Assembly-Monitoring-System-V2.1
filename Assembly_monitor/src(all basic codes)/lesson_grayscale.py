import cv2
camera = cv2.VideoCapture(0) #Open camera number 0

success, frame = camera.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert the frame to grayscale
print("Grayscale frame shape:", gray.shape)
print("Original shape:", frame.shape)
print("Color pixel:", frame[100,100])
print("Grayscale pixel:", gray[100,100])
camera.release()
