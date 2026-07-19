import cv2

for i in range(5):

    print(f"\nTesting camera index {i}")

    camera = cv2.VideoCapture(i)

    print("Opened:", camera.isOpened())

    success, frame = camera.read()

    print("Read success:", success)

    if success:
        print("FOUND CAMERA AT INDEX", i)

    camera.release()