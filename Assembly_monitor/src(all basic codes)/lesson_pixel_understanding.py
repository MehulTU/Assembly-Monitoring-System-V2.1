import cv2
camera = cv2.VideoCapture(1) #Open camera number 1
success, frame = camera.read() #Take a picture and save it to "frame"
print("frame shape:", frame.shape)
pixel = frame[100,100] #Get the pixel value at row 100, column 100
print("Pixel value at (100,100):", pixel)
camera.release()