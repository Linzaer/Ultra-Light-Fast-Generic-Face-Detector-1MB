# Original Author: Rahul Mangalampalli@https://www.kaggle.com/rahulmangalampalli/mafa-data
# Haojin Yang: extended to convert the MAFA dataset to the pascal voc compatible format.

from argparse import ArgumentParser
from os import path
import os

import cv2
import cv2 as cv
import hdf5storage
from tqdm import tqdm
import shutil
import xml.etree.cElementTree as ET

## Argparser
argparser = ArgumentParser()
argparser.add_argument("--mafa_root", type=str, default=None,
                       help="MAFA dataset root folder.")
argparser.add_argument("--train_mat", type=str, default=None,
                       help="The mat file contains the train labels.")
argparser.add_argument("--test_mat", type=str, default=None,
                       help="The mat file contains the test labels.")
argparser.add_argument("--export_dir", default=None, type=str,
                       help="Where the extracted images should be saved.")

##=============== All about the visualization of bboxes ==================#
def expand_box(square_box, scale_ratio=1.2):
    """Scale up the box"""
    assert (scale_ratio >= 1), "Scale ratio should be greater than 1."
    delta = int((square_box[2] - square_box[0]) * (scale_ratio - 1) / 2)
    left_x = square_box[0] - delta
    left_y = square_box[1] - delta
    right_x = square_box[2] + delta
    right_y = square_box[3] + delta
    return [left_x, left_y, right_x, right_y]


def fit_by_shifting(box, rows, cols):
    """Method 1: Try to move the box."""
    # Face box points.
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    # Check if moving is possible.
    if right_x - left_x <= cols and bottom_y - top_y <= rows:
        if left_x < 0:                  # left edge crossed, move right.
            right_x += abs(left_x)
            left_x = 0
        if right_x > cols:              # right edge crossed, move left.
            left_x -= (right_x - cols)
            right_x = cols
        if top_y < 0:                   # top edge crossed, move down.
            bottom_y += abs(top_y)
            top_y = 0
        if bottom_y > rows:             # bottom edge crossed, move up.
            top_y -= (bottom_y - rows)
            bottom_y = rows

    return [left_x, top_y, right_x, bottom_y]


def get_minimal_box(points):
    """
    Get the minimal bounding box of a group of points.
    The coordinates are also converted to int numbers.
    """
    min_x = int(min([point[0] for point in points]))
    max_x = int(max([point[0] for point in points]))
    min_y = int(min([point[1] for point in points]))
    max_y = int(max([point[1] for point in points]))
    return [min_x, min_y, max_x, max_y]


def points_in_box(points, box):
    """Check if box contains all the points"""
    minimal_box = get_minimal_box(points)
    return box[0] <= minimal_box[0] and \
        box[1] <= minimal_box[1] and \
        box[2] >= minimal_box[2] and \
        box[3] >= minimal_box[3]


def box_in_image(box, image):
    """Check if the box is in image"""
    rows = image.shape[0]
    cols = image.shape[1]
    return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows


def box_is_valid(image, points, box):
    """Check if box is valid."""
    # Box contains all the points.
    points_is_in_box = points_in_box(points, box)

    # Box is in image.
    box_is_in_image = box_in_image(box, image)

    # Box is square.
    w_equal_h = (box[2] - box[0]) == (box[3] - box[1])

    # Return the result.
    return box_is_in_image and points_is_in_box and w_equal_h



def fit_by_shrinking(box, rows, cols):
    """Method 2: Try to shrink the box."""
    # Face box points.
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    # The first step would be get the interlaced area.
    if left_x < 0:                  # left edge crossed, set zero.
        left_x = 0
    if right_x > cols:              # right edge crossed, set max.
        right_x = cols
    if top_y < 0:                   # top edge crossed, set zero.
        top_y = 0
    if bottom_y > rows:             # bottom edge crossed, set max.
        bottom_y = rows

    # Then found out which is larger: the width or height. This will
    # be used to decide in which dimension the size would be shrunken.
    width = right_x - left_x
    height = bottom_y - top_y
    delta = abs(width - height)
    # Find out which dimension should be altered.
    if width > height:                  # x should be altered.
        if left_x != 0 and right_x != cols:     # shrink from center.
            left_x += int(delta / 2)
            right_x -= int(delta / 2) + delta % 2
        elif left_x == 0:                       # shrink from right.
            right_x -= delta
        else:                                   # shrink from left.
            left_x += delta
    else:                               # y should be altered.
        if top_y != 0 and bottom_y != rows:     # shrink from center.
            top_y += int(delta / 2) + delta % 2
            bottom_y -= int(delta / 2)
        elif top_y == 0:                        # shrink from bottom.
            bottom_y -= delta
        else:                                   # shrink from top.
            top_y += delta

    return [left_x, top_y, right_x, bottom_y]


def fit_box(box, image: object, points: object):
    """
    Try to fit the box, make sure it satisfy following conditions:
    - A square.
    - Inside the image.
    - Contains all the points.
    If all above failed, return None.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    # First try to move the box.
    box_moved = fit_by_shifting(box, rows, cols)

    # If moving fails ,try to shrink.
    if box_is_valid(image, points, box_moved):
        return box_moved
    else:
        box_shrunken = fit_by_shrinking(box, rows, cols)

    # If shrink failed, return None
    if box_is_valid(image, points, box_shrunken):
        return box_shrunken

    # Finally, Worst situation.
    print("Fitting failed!")
    return None

##=============== END of All about the visualization of bboxes ==================#


##=============== All about the visualization of bboxes ==================#

def load_labels(label_file, is_train):
    out = hdf5storage.loadmat(label_file)
    samples = []
    if is_train:
        record = out['label_train'][0]
        for item in record:
            samples.append(
                {
                    'image_file': item[1][0],
                    'lables': [v for v in item[2].astype(int)]
                }
            )
    else:
        record = out['LabelTest'][0]
        for item in record:
            samples.append(
                {
                    'image_file': item[0][0],
                    'lables': [v for v in item[1].astype(int)]
                }
            )
    return samples


def parse_labels(raw_labels, is_train=True):
    """
    FOR TRAIN LABELS
        raw labels form: [x,y,w,h, x1,y1,x2,y2, x3,y3,w3,h3, occ_type, occ_degree,
        gender, race, orientation, x4,y4,w4,h4]
        (a) (x,y,w,h) is the bounding box of a face,
        (b) (x1,y1,x2,y2) is the position of two eyes.
        (c) (x3,y3,w3,h3) is the bounding box of the occluder. Note that (x3,y3)
            is related to the face bounding box position (x,y)
        (d) occ_type stands for the occluder type and has: 1 for simple, 2 for
            complex and 3 for human body.
        (e) occ_degree stands for the number of occluded face parts
        (f) gender and race stand for the gender and race of one face
        (g) orientation stands for the face orientation/pose, and has: 1-left,
            2-left frontal, 3-frontal, 4-right frontal, 5-right
        (h) (x4,y4,w4,h4) is the bounding box of the glasses and is set to
            (-1,-1,-1,-1) when no glasses. Note that (x4,y4) is related to the
            face bounding box position (x,y)

    FOR TEST LABELS
        The format is stored in a 18d array (x,y,w,h,face_type,x1,y1,w1,h1, occ_type,
        occ_degree, gender, race, orientation, x2,y2,w2,h2), where
        (a) (x,y,w,h) is the bounding box of a face, 
        (b) face_type stands for the face type and has: 1 for masked face, 2 for
            unmasked face and 3 for invalid face.
        (c) (x1,y1,w1,h1) is the bounding box of the occluder. Note that (x1,y1)
            is related to the face bounding box position (x,y)
        (d) occ_type stands for the occluder type and has: 1 for simple, 2 for 
            complex and 3 for human body.
        (e) occ_degree stands for the number of occluded face parts
        (f) gender and race stand for the gender and race of one face
        (g) orientation stands for the face orientation/pose, and has: 1-left, 
            2-left frontal, 3-frontal, 4-right frontal, 5-right
        (h) (x2,y2,w2,h2) is the bounding box of the glasses and is set to 
            (-1,-1,-1,-1) when no glasses.  Note that (x2,y2) is related to the 
            face bounding box position (x,y)

    """
    labels = []

    # For the purpose of creating a dataset benchmark for masked face detection, we will exclude samples with a
    # occlude degree lower than 3. We also exclude the samples with human body (e.g., hands) occlusion.

    if is_train:
        for raw_label in raw_labels:
            # filtering out the less occluded faces
            if raw_label[13] < 3 or raw_label[12] == 3:
                continue

            labels.append(
                {
                    'face': [raw_label[0], raw_label[1], raw_label[2], raw_label[3]],
                    'eyes': [raw_label[4], raw_label[5], raw_label[6], raw_label[7]],
                    'occlude': {
                        'location': [raw_label[8], raw_label[9], raw_label[10], raw_label[11]],
                        'type': raw_label[12],
                        'degree': raw_label[13]},
                    'gender': raw_label[14],
                    'race': raw_label[15],
                    'orientation': raw_label[16],
                    'glass': [raw_label[17], raw_label[18], raw_label[19], raw_label[20]]
                }
            )
    else:
        for raw_label in raw_labels:
            # filtering out the less occluded faces and non-occluded faces
            if raw_label[4] != 1 or raw_label[10] < 3 or raw_label[9] == 3:
                continue

            labels.append(
                {
                    'face': [raw_label[0], raw_label[1], raw_label[2], raw_label[3]],
                    'face_type': raw_label[4],
                    'occlude': {
                        'location': [raw_label[5], raw_label[6], raw_label[7], raw_label[8]],
                        'type': raw_label[9], # exclude human body
                        'degree': raw_label[10]}, # degree 1 and 2 not acceptable
                    'gender': raw_label[11],
                    'race': raw_label[12],
                    'orientation': raw_label[13],
                    'glass': [raw_label[14], raw_label[15], raw_label[16], raw_label[17]]
                }
            )

    return labels


def draw_face(image, labels, color=(0, 255, 0)):
    for label in labels:
        x, y, w, h = label['face']
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)


def draw_mask(image, labels, color=(0, 0, 255)):
    for label in labels:
        x, y, w, h = label['face']
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        _x, _y, _w, _h = label['occlude']['location']
        _x =_x+x
        _y=_y+y
        cv.rectangle(image, (_x, _y), (_x + _w, _y + _h), (0, 0, 255), 3)


def export_face(image, labels, export_file, occ_types=[1, 2, 3], min_size=120, export_size=112):
    """
    Export face areas in an image.
    Args:
        image: the image as a numpy array.
        labels: MAFA labels.
        export_file: the output file name. If more than one face exported, a subfix 
            number will be appended to the file name.
        occ_types: a list of occlusion type which should be exported.
        min_size: the minimal size of faces should be exported.
        exprot_size: the output size of the square image.
    Returns:
        the exported image, or None.
    """
    # Crop the face
    idx_for_face = 0
    image_faces = []
    for label in labels:
        # Not all faces in label is occluded. Filter the image by occlusion,
        # size, etc.
        x, y, w, h = label['face']
        if w < min_size or h < min_size:
            continue

        if label['occlude']['type'] not in occ_types:
            continue

        # Enlarge the face area and make it a square.
        box = expand_box([x, y, x+w, y+h], 1.3)
        box = fit_box(box, image, [(box[0], box[1]), (box[2], box[3])])
        if box is not None:
            image_face = image[box[1]:box[3], box[0]:box[2]]
        else:
            return None

        # Resize and save image.
        image_face = cv2.resize(image_face, (export_size, export_size))
        new_file = export_file.rstrip(
            '.jpg') + '-{}.jpg'.format(idx_for_face)
        cv2.imwrite(new_file, image_face)
        image_faces.append(image_face)

        idx_for_face += 1

    return image_faces


def write_voc_style_ann(labels, img_file_name, num_human_occ):
    # example of pascal voc annotation
    # <object>
    # <name>face</name>
    # <pose>Unspecified</pose>
    # <truncated>1</truncated>
    # <difficult>0</difficult>
    # <bndbox>
    # <xmin>146</xmin>
    # <ymin>215</ymin>
    # <xmax>179</xmax>
    # <ymax>248</ymax>
    # </bndbox>
    # <has_lm>0</has_lm>
    # </object>

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "MAFA"
    ET.SubElement(root, "filename").text = img_file_name
    ET.SubElement(root, "segmented").text = '0'

    for label in labels:
        obj = ET.SubElement(root, "object")
        # we label the human body occlusion as the 'face', since our purpose is to build a dataset of face/masked_face.
        # Human body occlusion should not be considered as masked_face.
        occ_type = label['occlude']['type']
        if occ_type == 3: # human body
            ET.SubElement(obj, "name").text = 'face'
            num_human_occ +=1
        else:
            ET.SubElement(obj, "name").text = 'masked_face'
        ET.SubElement(obj, "pose").text = 'Unspecified'
        ET.SubElement(obj, "truncated").text = '1'
        ET.SubElement(obj, "difficult").text = '0'
        ET.SubElement(obj, "has_lm").text = '0'

        x, y, w, h = label['face']
        bnbbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnbbox, "xmin").text = str(x)
        ET.SubElement(bnbbox, "ymin").text = str(y)
        ET.SubElement(bnbbox, "xmax").text = str(x+w)
        ET.SubElement(bnbbox, "ymax").text = str(y+h)

        _x, _y, _w, _h = label['occlude']['location']
        _x = _x + x
        _y = _y + y
        occluder = ET.SubElement(obj, "occluder")
        ET.SubElement(occluder, "xmin").text = str(_x)
        ET.SubElement(occluder, "ymin").text = str(_y)
        ET.SubElement(occluder, "xmax").text = str(_x+_w)
        ET.SubElement(occluder, "ymax").text = str(_y+_h)

        # We could further extend the annotations using more information such as glasses and eyes etc.
    tree = ET.ElementTree(root)
    return tree, num_human_occ


if __name__ == "__main__":
    # Get all args.
    args = argparser.parse_args()

    # is_train = False
    is_train = True

    img_dir = 'train-images' if is_train else 'test-images'
    mat = args.train_mat if is_train else args.test_mat

    # Load annotations from the mat file.
    samples = load_labels(mat, is_train=is_train)

    TARGET_SAMPLES = 15542
    num = 0
    num_human_occ = 0

    # Extract the face images.
    if args.export_dir is not None:
        export_face_image_dir = path.join(args.export_dir, 'JPEGImages')
        export_imageSets_dir = path.join(args.export_dir, 'ImageSets')
        main_imageSets_dir = path.join(export_imageSets_dir, 'Main')
        export_annotations_dir = path.join(args.export_dir, 'Annotations')
        if not os.path.exists(export_face_image_dir):
            os.makedirs(export_face_image_dir)
        if not os.path.exists(export_imageSets_dir):
            os.makedirs(export_imageSets_dir)
            if not os.path.exists(main_imageSets_dir):
                os.makedirs(main_imageSets_dir)
        if not os.path.exists(export_annotations_dir):
            os.makedirs(export_annotations_dir)

    # create image set list files for train and test
    listfile = None
    if is_train:
        listfile = path.join(main_imageSets_dir, 'trainval.txt')
    else:
        listfile = path.join(main_imageSets_dir, 'test.txt')
    with open(listfile, 'w') as f:
        # loop through all the annotations and do processing. Here we are going to
        # extract all the occluded faces and save them in a new image file.
        for sample in tqdm(samples):
            labels = parse_labels(sample['lables'], is_train=is_train)
            if len(labels) == 0:
                continue

            img_file_name = sample['image_file']
            img_url = path.join(args.mafa_root, img_dir, img_file_name)
            image = cv2.imread(img_url)
            assert not isinstance(image, type(None)), 'image not found'

            # We could use the following method to visually check the annotation quality.
            # draw_face(image, labels)
            draw_mask(image, labels)

            # write the image name without extension
            img_name_wo_ext = os.path.splitext(img_file_name)[0]
            # only for creating image name of human body split
            # img_name_wo_ext = img_name_wo_ext + '_human_occ'

            f.write(img_name_wo_ext)
            f.write('\n')

            # create xml annotation file
            xml_annotation_file = path.join(export_annotations_dir, img_name_wo_ext + '.xml')
            f_xml, num_human_occ = write_voc_style_ann(labels, img_file_name, num_human_occ)
            f_xml.write(xml_annotation_file)

            # Extract the face images.
            if args.export_dir is not None:
                export_face_image_file = path.join(export_face_image_dir, img_name_wo_ext + '.jpg')
                # copy images
                shutil.copyfile(img_url, export_face_image_file)

                # visualize faces for validation purpose.
                # exported_face = export_face(image, labels, export_face_image_file, occ_types=[1, 2, 3], min_size=60, export_size=112)
            num += 1
            if num == TARGET_SAMPLES:
                break
    f.close()
    print('{} human body occluded faces'.format(num_human_occ))

