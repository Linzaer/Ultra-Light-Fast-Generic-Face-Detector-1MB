#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import shutil
from xml.dom.minidom import Document

import cv2

rootdir = "./wider_face_add_lm_10_10"
os.makedirs(rootdir)
retinaface_gt_file_path = "./retinaface_labels/"
convet2yoloformat = False
convert2vocformat = True
resized_dim = (48, 48)

minsize2select = 10  # min face size
usepadding = True

datasetprefix = "./wider_face"  #


def convertimgset(img_set="train"):
    imgdir = datasetprefix + "/WIDER_" + img_set + "/images"
    gtfilepath = retinaface_gt_file_path + img_set + "/label.txt"
    imagesdir = rootdir + "/JPEGImages"
    vocannotationdir = rootdir + "/Annotations"
    labelsdir = rootdir + "/labels"
    if not os.path.exists(imagesdir):
        os.mkdir(imagesdir)
    if not os.path.exists(rootdir + "/ImageSets"):
        os.mkdir(rootdir + "/ImageSets")
    if not os.path.exists(rootdir + "/ImageSets/Main"):
        os.mkdir(rootdir + "/ImageSets/Main")

    if convet2yoloformat:
        if not os.path.exists(labelsdir):
            os.mkdir(labelsdir)
    if convert2vocformat:
        if not os.path.exists(vocannotationdir):
            os.mkdir(vocannotationdir)
    index = 0

    f_set = open(rootdir + "/ImageSets/Main/" + img_set + ".txt", 'w')
    current_filename = ""
    bboxes = []
    lms = []
    with open(gtfilepath, 'r') as gtfile:
        while True:  # and len(faces)<10
            line = gtfile.readline().strip()
            if line == "":
                if len(bboxes) != 0:
                    method_name(bboxes, filename, saveimg, vocannotationdir, lms, img_set)
                    cv2.imwrite(imagesdir + "/" + filename, saveimg)
                    imgfilepath = filename[:-4]
                    f_set.write(imgfilepath + '\n')
                    print("end!")
                break
            if line.startswith("#"):
                if index != 0 and convert2vocformat:
                    if len(bboxes) != 0:
                        method_name(bboxes, filename, saveimg, vocannotationdir, lms, img_set)
                        cv2.imwrite(imagesdir + "/" + filename, saveimg)
                        imgfilepath = filename[:-4]
                        f_set.write(imgfilepath + '\n')
                    else:
                        print("no face")

                current_filename = filename = line[1:].strip()
                print(("\r" + str(index) + ":" + filename + "\t\t\t"))
                index = index + 1
                bboxes = []
                lms = []
                continue
            else:
                imgpath = imgdir + "/" + current_filename
                img = cv2.imread(imgpath)
                if not img.data:
                    break
                saveimg = img.copy()
                showimg = saveimg.copy()
                line = [float(x) for x in line.strip().split()]
                if int(line[3]) <= 0 or int(line[2]) <= 0:
                    continue
                x = int(line[0])
                y = int(line[1])
                width = int(line[2])
                height = int(line[3])
                bbox = (x, y, width, height)
                x2 = x + width
                y2 = y + height
                if width >= minsize2select and height >= minsize2select:
                    bboxes.append(bbox)
                    if img_set == "train":
                        if line[4] == -1:
                            lms.append(-1)
                        else:
                            lm = []
                            for i in range(5):
                                x = line[4 + 3 * i]
                                y = line[4 + 3 * i + 1]
                                lm.append((x, y))
                            lm.append(int(line[4 + 3 * i + 2]))
                            lm.append(line[19])
                            lms.append(lm)
                    cv2.rectangle(showimg, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0))
                else:
                    saveimg[y:y2, x:x2, :] = (104, 117, 123)
                    cv2.rectangle(showimg, (x, y), (x2, y2), (0, 0, 255))
                filename = filename.replace("/", "_")

                if convet2yoloformat:
                    height = saveimg.shape[0]
                    width = saveimg.shape[1]
                    txtpath = labelsdir + "/" + filename
                    txtpath = txtpath[:-3] + "txt"
                    ftxt = open(txtpath, 'w')
                    for i in range(len(bboxes)):
                        bbox = bboxes[i]
                        xcenter = (bbox[0] + bbox[2] * 0.5) / width
                        ycenter = (bbox[1] + bbox[3] * 0.5) / height
                        wr = bbox[2] * 1.0 / width
                        hr = bbox[3] * 1.0 / height
                        txtline = "0 " + str(xcenter) + " " + str(ycenter) + " " + str(wr) + " " + str(hr) + "\n"
                        ftxt.write(txtline)
                    ftxt.close()
    f_set.close()


def method_name(bboxes, filename, saveimg, vocannotationdir, lms, img_set):
    xmlpath = vocannotationdir + "/" + filename
    xmlpath = xmlpath[:-3] + "xml"
    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('widerface')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)
    filenamenode = doc.createElement('filename')
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)
    source = doc.createElement('source')
    annotation.appendChild(source)
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('wider face Database'))
    source.appendChild(database)
    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)
    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)
    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('-1'))
    source.appendChild(flickrid)
    owner = doc.createElement('owner')
    annotation.appendChild(owner)
    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('yanyu'))
    owner.appendChild(flickrid_o)
    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('yanyu'))
    owner.appendChild(name_o)
    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(saveimg.shape[1])))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(saveimg.shape[0])))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(saveimg.shape[2])))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode('face'))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[0] + bbox[2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[1] + bbox[3])))
        bndbox.appendChild(ymax)

        if img_set == "train":
            has_lm = doc.createElement('has_lm')

            if lms[i] == -1:
                has_lm.appendChild(doc.createTextNode('0'))
            else:
                has_lm.appendChild(doc.createTextNode('1'))
                lm = doc.createElement('lm')
                objects.appendChild(lm)

                x1 = doc.createElement('x1')
                x1.appendChild(doc.createTextNode(str(lms[i][0][0])))
                lm.appendChild(x1)

                y1 = doc.createElement('y1')
                y1.appendChild(doc.createTextNode(str(lms[i][0][1])))
                lm.appendChild(y1)

                x2 = doc.createElement('x2')
                x2.appendChild(doc.createTextNode(str(lms[i][1][0])))
                lm.appendChild(x2)

                y2 = doc.createElement('y2')
                y2.appendChild(doc.createTextNode(str(lms[i][1][1])))
                lm.appendChild(y2)

                x3 = doc.createElement('x3')
                x3.appendChild(doc.createTextNode(str(lms[i][2][0])))
                lm.appendChild(x3)

                y3 = doc.createElement('y3')
                y3.appendChild(doc.createTextNode(str(lms[i][2][1])))
                lm.appendChild(y3)

                x4 = doc.createElement('x4')
                x4.appendChild(doc.createTextNode(str(lms[i][3][0])))
                lm.appendChild(x4)

                y4 = doc.createElement('y4')
                y4.appendChild(doc.createTextNode(str(lms[i][3][1])))
                lm.appendChild(y4)

                x5 = doc.createElement('x5')
                x5.appendChild(doc.createTextNode(str(lms[i][4][0])))
                lm.appendChild(x5)

                y5 = doc.createElement('y5')
                y5.appendChild(doc.createTextNode(str(lms[i][4][1])))
                lm.appendChild(y5)

                visible = doc.createElement('visible')
                visible.appendChild(doc.createTextNode(str(lms[i][5])))
                lm.appendChild(visible)

                blur = doc.createElement('blur')
                blur.appendChild(doc.createTextNode(str(lms[i][6])))
                lm.appendChild(blur)
            objects.appendChild(has_lm)
    f = open(xmlpath, "w")
    f.write(doc.toprettyxml(indent=''))
    f.close()


def generatetxt(img_set="train"):
    gtfilepath = rootdir + "/wider_face_split/wider_face_" + img_set + "_bbx_gt.txt"
    f = open(rootdir + "/" + img_set + ".txt", "w")
    with open(gtfilepath, 'r') as gtfile:
        while (True):  # and len(faces)<10
            filename = gtfile.readline()[:-1]
            if (filename == ""):
                break
            filename = filename.replace("/", "_")
            imgfilepath = datasetprefix + "/images/" + filename
            f.write(imgfilepath + '\n')
            numbbox = int(gtfile.readline())
            for i in range(numbbox):
                line = gtfile.readline()
    f.close()


def generatevocsets(img_set="train"):
    if not os.path.exists(rootdir + "/ImageSets"):
        os.mkdir(rootdir + "/ImageSets")
    if not os.path.exists(rootdir + "/ImageSets/Main"):
        os.mkdir(rootdir + "/ImageSets/Main")
    gtfilepath = rootdir + "/wider_face_split/wider_face_" + img_set + "_bbx_gt.txt"
    f = open(rootdir + "/ImageSets/Main/" + img_set + ".txt", 'w')
    with open(gtfilepath, 'r') as gtfile:
        while (True):  # and len(faces)<10
            filename = gtfile.readline()[:-1]
            if (filename == ""):
                break
            filename = filename.replace("/", "_")
            imgfilepath = filename[:-4]
            f.write(imgfilepath + '\n')
            numbbox = int(gtfile.readline())
            for i in range(numbbox):
                line = gtfile.readline()
    f.close()


def convertdataset():
    img_sets = ["train", "val"]
    for img_set in img_sets:
        convertimgset(img_set)


if __name__ == "__main__":
    convertdataset()
    shutil.move(rootdir + "/ImageSets/Main/" + "train.txt", rootdir + "/ImageSets/Main/" + "trainval.txt")
    shutil.move(rootdir + "/ImageSets/Main/" + "val.txt", rootdir + "/ImageSets/Main/" + "test.txt")
