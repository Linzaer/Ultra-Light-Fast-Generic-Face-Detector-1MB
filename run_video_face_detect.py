"""
This code uses the pytorch model to detect faces from live video or camera.
"""

import sys
import cv2

from vision.ssd.config.fd_config import define_img_size

input_img_size = 320  # define input size ,default optional(128/160/320/480/640/1280)
define_img_size(input_img_size)

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

label_path = "./models/voc-model-labels.txt"

# net_type = "mb_tiny_fd"          # inference faster,lower precision
net_type = "mb_tiny_RFB_fd"  # inference lower,higher precision

cap = cv2.VideoCapture("/home/linzai/Videos/video/16_4.MP4")  # capture from video
# cap = cv2.VideoCapture(0)  # capture from camera

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = "cuda:0"

candidate_size = 500
threshold = 0.7

if net_type == 'mb_tiny_fd':
    model_path = "models/pretrained/Mb_Tiny_FD_train_input_320.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'mb_tiny_RFB_fd':
    model_path = "models/pretrained/Mb_Tiny_RFB_FD_train_input_320.pth"
    # model_path = "models/pretrained/Mb_Tiny_RFB_FD_train_input_640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

timer = Timer()
sum = 0
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        print("end")
        break
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
    interval = timer.end()
    print('Time: {:.6f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f" {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (0, 0, 255),
                    2)  # line type
    orig_image = cv2.resize(orig_image, None, None, fx=0.8, fy=0.8)
    sum += boxes.size(0)
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("all face num:{}".format(sum))
