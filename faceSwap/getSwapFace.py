#! /usr/bin/env python
import os
import cv2
import argparse

from .face_detection import select_face
from .face_swap import face_swap

class getSwapFace(object):
    def __init__(self,src_img, dst_img, sav_img):
        # Read images
        self.src_img = cv2.imread(src_img)
        self.dst_img = cv2.imread(dst_img)
        self.sav_img = sav_img

        # Select src face
        self.src_points, self.src_shape, self.src_face = select_face(self.src_img)
        # Select dst face
        self.dst_points, self.dst_shape, self.dst_face = select_face(self.dst_img)

    def getResult(self,args):
        output = face_swap(self.src_face, self.dst_face, self.src_points, self.dst_points, self.dst_shape, self.dst_img, args)
        # dir_path = os.path.dirname(self.sav_img)
        # if not os.path.isdir(dir_path):
        #     os.makedirs(dir_path)

        cv2.imwrite(self.sav_img, output)
        return self.sav_img

if __name__ == "__main__":
    model = getSwapFace("testliu2.jpg","test1.jpg","liuliu.jpg")
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
    parser.add_argument('--correct_color', default=False, action='store_true', help='Correct color')
    parser.add_argument('--no_debug_window', default=False, action='store_true', help='Don\'t show debug window')
    args = parser.parse_args()
    model.getResult(args)

