# -*- coding: utf-8 -*-
# Image Extraction Test
# copyright 2015 Chibwe Ltd

# vectorise test images.

import cv2
import os
from glob import glob
import fnmatch
import numpy as np
import csv
import cPickle


## Display an image
def DisplayImg(img, scale=1):
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    while (1):
        cv2.imshow('roi', img)
##        cv2.setMouseCallback("in", CaptureMouse,img)
        k = cv2.waitKey(33)
        if k == 27:    # Esc key to stop
            cv2.destroyAllWindows()
            break
        elif k == -1:  # normally -1 returned,so don't print it
            continue
        else:
            print k  # else print its value


def get_files(path):
    matches = []
    files = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            # print filename
            files.append(filename)
            matches.append(os.path.join(root, filename))

    files = np.array(files)
    file_info = np.unique(files, return_counts=True)
    return files


def verify_images(data, path):
    exist_list = []       #list of existing and valid image with full path in local director
    code_labels = []    #gender labels for existing and valid images
    non_exist_list = []   #list of non-existing images
    img_list = []         #resized pixel data for existing and valid images
    for img in data:
        client = img[0]+'/'
        site = img[1]+'/'
        entrance = img[2]+'/'
        date = img[3]+'/'
        hour = img[4]+'/'
        fileid = img[5]

        img_path = path+client+site+entrance+date+hour+fileid
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # img=cv2.resize(img, (40,60), interpolation = cv2.INTER_CUBIC)
                img = cv2.resize(img, (40, 60), interpolation=cv2.INTER_CUBIC)
                img_list.append(img)
                exist_list.append(img_path)
                code_labels.append(0)
            # finish creating label information for existing and valid images

            else:
                print "broken image at %s" % img_path
            #  append label value to lable list
        else:
            non_exist_list.append(img_path)

    img_list = np.array(img_list)
    exist_list = np.array(exist_list)
    non_exist_list = np.array(non_exist_list)
    code_labels = np.array(code_labels)

    return img_list, exist_list, non_exist_list, code_labels


def load_image_data(exist_list):
    imglist = []
    # print "loading %s" %path
    # filelist=os.listdir(path)
    # for f in filelist:
    for index in xrange(1, len(exist_list)):
        # print 'loading image id %d' %index
        img_path = exist_list[index]

        if os.path.isfile(img_path):

            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (40, 60), interpolation=cv2.INTER_CUBIC)
                imglist.append(img)
            else:
                print "none type image: %s" % img_path
        else:
            print "cannot load image %s" % img_path
    return np.array(imglist)


def pickle_data(path, data):
    file = path+'test_data.pkl'
    save_file = open(file, 'wb')
    cPickle.dump(data, save_file, -1)
    save_file.close()


if __name__ == '__main__':

    # The following path parameters need to be changed according in the ec2 instance
    project_path = os.path.dirname(__file__)
    path = os.path.dirname(__file__)

    img_data_path = "/srv/secureimage/test_images/"
    #project_path = "c:/users/xie/playground/cctv classification/"

    # files=get_files(path)
    img_csv = img_data_path + 'summary.csv'

    data_sum = np.genfromtxt(img_csv, dtype=None, delimiter=',', skip_header=1)
  
    total_data = data_sum

    img_list, exist_list, non_exist_list = verify_images(total_data, img_data_path)
    extracted_data = [img_list, exist_list, non_exist_list, code_labels]

    pickle_folder = project_path+"test_data/"
    pickle_data(pickle_folder, extracted_data)
