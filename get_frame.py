import cv2

cap = cv2.VideoCapture('test.mp4')  # capture from video

i = 23
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        print("end")
        break
    path = "../test_img/" + str(i) + ".jpg"
    cv2.imwrite(path, orig_image)
    print(i)
    i += 1
