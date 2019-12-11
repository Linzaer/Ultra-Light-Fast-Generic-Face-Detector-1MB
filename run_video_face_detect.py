"""
This code uses the pytorch model to detect faces from live video or camera.
"""
import argparse
import sys
import cv2

from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser(description='detect_video')

parser.add_argument(
    '--net_type',
    default="RFB",
    type=str,
    help=
    'The network architecture ,optional: RFB (higher precision) or slim (faster)'
)
parser.add_argument(
    '--input_size',
    default=480,
    type=int,
    help=
    'define network input size,default optional value 128/160/320/480/640/1280'
)
parser.add_argument('--threshold',
                    default=0.7,
                    type=float,
                    help='score threshold')
parser.add_argument('--candidate_size',
                    default=1000,
                    type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str, help='imgs dir')
parser.add_argument('--test_device',
                    default="cuda:0",
                    type=str,
                    help='cuda:0 or cpu')
parser.add_argument('--video_path',
                    default="test.mp4",
                    type=str,
                    help='path of video')
args = parser.parse_args()

input_img_size = args.input_size
define_img_size(
    input_img_size
)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

label_path = "./models/train-version-slim/voc-model-labels.txt"

net_type = args.net_type

cap = cv2.VideoCapture(args.video_path)  # capture from video
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('out4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              size)
# cap = cv2.VideoCapture(0)  # capture from camera

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = args.test_device

candidate_size = args.candidate_size
threshold = args.threshold

if net_type == 'slim':
    model_path = "models/train-version-slim/slim-Epoch-360-Loss-2.6601163205646334.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net,
                                            candidate_size=candidate_size,
                                            device=test_device)
elif net_type == 'RFB':
    model_path = "models/train-version-RFB/RFB-Epoch-420-Loss-2.635296208517892.pth"
    # model_path = "models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names),
                                is_test=True,
                                device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net,
                                                candidate_size=candidate_size,
                                                device=test_device)
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
    boxes, labels, probs = predictor.predict(image, candidate_size / 2,
                                             threshold)
    interval = timer.end()
    print('Time: {:.6f}s, Detect Objects: {:d}.'.format(
        interval, labels.size(0)))
    fps = 'fps: %d' % (1 / interval)
    cv2.putText(
        orig_image,
        fps,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # font scale
        (0, 255, 242),
        2)  # line type
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        if max(box[2] - box[0], box[3] - box[1]) < 100:  # TODO: 过滤掉较小的bbox
            continue
        # label = f" {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]),
                      (0, 255, 0), 1)

        cv2.putText(
            orig_image,
            str(probs[i]),
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # font scale
            (0, 255, 0),
            1)  # line type

    # orig_image = cv2.resize(orig_image, None, None, fx=0.8, fy=0.8)
    sum += boxes.size(0)
    # cv2.imshow('annotated', orig_image)
    videoWriter.write(orig_image)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    # break
cap.release()
cv2.destroyAllWindows()
print("all face num:{}".format(sum))
