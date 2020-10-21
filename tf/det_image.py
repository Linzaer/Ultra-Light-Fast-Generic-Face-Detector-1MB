import argparse
import sys

import cv2
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(
    description='convert model')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--img_path', default='imgs/test_input.jpg', type=str,
                    help='Image path for inference')
args = parser.parse_args()


def main():
    if args.net_type == 'slim':
        model_path = "export_models/slim/"
    elif args.net_type == 'RFB':
        model_path = "export_models/RFB/"
    else:
        print("The net type is wrong!")
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)

    img = cv2.imread(args.img_path)
    h, w, _ = img.shape
    img_resize = cv2.resize(img, (320, 240))
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img_resize = img_resize - 127.0
    img_resize = img_resize / 128.0

    results = model.predict(np.expand_dims(img_resize, axis=0))  # result=[background,face,x1,y1,x2,y2]

    for result in results:
        start_x = int(result[2] * w)
        start_y = int(result[3] * h)
        end_x = int(result[4] * w)
        end_y = int(result[5] * h)

        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    cv2.imwrite(f'imgs/test_output_{args.net_type}.jpg', img)


if __name__ == '__main__':
    main()
