import argparse
import cv2
import time

from TFLiteFaceDetector import UltraLightFaceDetecion


parser = argparse.ArgumentParser(description='TFLite Face Detector')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--img_path', type=str, help='Image path for inference')
parser.add_argument('--video_path', type=str, help='Video path for inference')

args = parser.parse_args()


def image_inference(image_path, model_path, color=(125, 255, 0)):

    fd = UltraLightFaceDetecion(model_path,
                                conf_threshold=0.6)

    img = cv2.imread(image_path)

    boxes, scores = fd.inference(img)

    for result in boxes.astype(int):
        cv2.rectangle(img, (result[0], result[1]),
                      (result[2], result[3]), color, 2)

    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_inference(video, model_path, color=(125, 255, 0)):

    fd = UltraLightFaceDetecion(model_path,
                                conf_threshold=0.88)

    cap = cv2.VideoCapture(video)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        start_time = time.perf_counter()
        boxes, scores = fd.inference(frame)
        print(time.perf_counter() - start_time)

        for result in boxes.astype(int):
            cv2.rectangle(frame, (result[0], result[1]),
                          (result[2], result[3]), color, 2)

        cv2.imshow('res', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    filepath = f"pretrained/version-{args.net_type}-320_without_postprocessing.tflite"

    if args.img_path:
        image_inference(args.img_path, filepath)
    elif args.video_path:
        video_inference(args.video_path, filepath)
    else:
        print('--ima_path or --video_path must be filled')
