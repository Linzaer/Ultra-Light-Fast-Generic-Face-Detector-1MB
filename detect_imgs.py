"""
This code is used to batch detect images in a folder.
"""
import os
import sys

import cv2

from vision.ssd.config.fd_config import define_img_size

input_img_size = 640  # define input size ,default optional(128/160/320/480/640/1280)
define_img_size(input_img_size)

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

# net_type = "mb_tiny_fd"          # inference faster,lower precision
net_type = "mb_tiny_RFB_fd"  # inference lower,higher precision

path = "imgs"
result_path = "./detect_imgs_results"
label_path = "./models/voc-model-labels.txt"
test_device = "cuda:0"

candidate_size = 1500
threshold = 0.7

class_names = [name.strip() for name in open(label_path).readlines()]
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

if not os.path.exists(result_path):
    os.makedirs(result_path)
listdir = os.listdir(path)
sum = 0
for file_path in listdir:
    img_path = os.path.join(path, file_path)
    orig_image = cv2.imread(img_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
    sum += boxes.size(0)
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{probs[i]:.2f}"
        # cv2.putText(orig_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(orig_image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(result_path, file_path), orig_image)
    print(f"Found {len(probs)} faces. The output image is {path}")
print(sum)
