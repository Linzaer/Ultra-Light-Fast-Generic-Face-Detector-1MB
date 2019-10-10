"""
This code is used to check the data size distribution in the dataset.
"""
import xml.etree.ElementTree as ET
from math import sqrt as sqrt

import cv2
import matplotlib.pyplot as plt

# sets = [("./data/wider_face_add_lm_10_10", "trainval")]
sets = [("./data/wider_face_add_lm_10_10", "test")]

classes = ['face']

if __name__ == '__main__':
    width = []
    height = []

    for image_set, set in sets:
        image_ids = open('{}/ImageSets/Main/{}.txt'.format(image_set, set)).read().strip().split()
        for image_id in image_ids:
            img_path = '{}/JPEGImages/{}.jpg'.format(image_set, image_id)
            label_file = open('{}/Annotations/{}.xml'.format(image_set, image_id))
            tree = ET.parse(label_file)
            root = tree.getroot()
            size = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)
            img = cv2.imread(img_path)
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 2:
                    continue
                cls_id = classes.index(cls)

                xmlbox = obj.find('bndbox')
                xmin = int(xmlbox.find('xmin').text)
                ymin = int(xmlbox.find('ymin').text)
                xmax = int(xmlbox.find('xmax').text)
                ymax = int(xmlbox.find('ymax').text)
                w = xmax - xmin
                h = ymax - ymin

                # img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 8)
                w_change = (w / img_w) * 320
                h_change = (h / img_h) * 240
                s = w_change * h_change
                if w_change / h_change > 6:
                    print("{}/{}/{}/{}".format(xmin, xmax, ymin, ymax))
                width.append(sqrt(s))
                height.append(w_change / h_change)
            print(img_path)
            # img = cv2.resize(img, (608, 608))
            # cv2.imwrite('{}_{}'.format(image_set.split('/')[-1], set), img)
            # cv2.waitKey()

    plt.plot(width, height, 'ro')
    plt.show()
