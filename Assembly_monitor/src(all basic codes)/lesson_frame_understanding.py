import cv2
camera = cv2.VideoCapture(1) #Open camera number 1
print("Camera opened:", camera.isOpened())

success, frame = camera.read() #Take a picture and save it to "frame"
print("Frame read success:", frame.shape)

camera.release()