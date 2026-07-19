import cv2
camera = cv2.VideoCapture(1) #Open camera number 1
print("press Enter to capture an image")
input() #Wait for the user to press Enter

success, frame = camera.read() #Take a picture and save it to "frame"

gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

print("Move your hand into the camera")
print("press Enter to capture another image")
input() #Wait for the user to press Enter

success, frame2 = camera.read() #Take another picture and save it to "frame2"
gray2= cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

difference = cv2.absdiff(gray1, gray2) #Calculate the absolute difference between the two grayscale images

motion_value = difference.sum() #Sum all the pixel values in the difference image to get a motion value
print("Motion Value:", motion_value)
camera.release()