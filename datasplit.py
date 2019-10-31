#!/usr/bin/env python3

import os
import glob
import sys
import shutil
import random


def main():
    SOURCE_DIR = os.path.expanduser('~/group/msuzuki/MVTechAD')
    DST_DIR = os.path.join(os.path.expanduser(os.path.dirname(os.path.abspath(__file__))), os.path.basename(sys.argv[1]))
    os.mkdir(DST_DIR)
    print(SOURCE_DIR, DST_DIR)

    # Category
    category = [cat for cat in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, cat))]
    print("Category", category)

    for cat in category:
        trainsrcdir = os.path.join(SOURCE_DIR, cat, 'train')
        traindstdir = os.path.join(DST_DIR, cat, 'train')

        testsrcdir = os.path.join(SOURCE_DIR, cat, 'test')
        testdstdir = os.path.join(DST_DIR, cat, 'test')
        os.makedirs(testdstdir)

        print("# Copy Train")
        shutil.copytree(trainsrcdir, traindstdir)

        print("# Copy Test good")
        shutil.copytree(os.path.join(testsrcdir, 'good'), os.path.join(testdstdir, 'good'))

        print("# Split Test defect")
        defect_category = [defcat for defcat in os.listdir(testsrcdir) if (os.path.isdir(os.path.join(testsrcdir, defcat)) and not 'good' == os.path.basename(defcat))]
        print(defect_category)

        defcatdstdirtest = os.path.join(testdstdir, 'defective')
        defcatdstdirtrain = os.path.join(traindstdir, 'defective')
        os.makedirs(defcatdstdirtest)
        os.makedirs(defcatdstdirtrain)
        for defcat in defect_category:
            defcatsrcdir = os.path.join(testsrcdir, defcat)
            print("# random split")
            defcat_imgs = glob.glob(os.path.join(defcatsrcdir, '*.png'))
            random.shuffle(defcat_imgs)
            defcat_imgs_train = defcat_imgs[:int(len(defcat_imgs)/2)]
            defcat_imgs_test = defcat_imgs[int(len(defcat_imgs)/2):]
            print("# copy")
            for img in defcat_imgs_train:
                shutil.copyfile(img, os.path.join(defcatdstdirtrain, defcat+os.path.basename(img)))
            for img in defcat_imgs_test:
                shutil.copyfile(img, os.path.join(defcatdstdirtest, defcat+os.path.basename(img)))

        print("#Copy Train to Test")
        shutil.copytree(os.path.join(traindstdir, 'good'), os.path.join(testdstdir, 'train_good'))
        shutil.copytree(os.path.join(traindstdir, 'defective'), os.path.join(testdstdir, 'train_defective'))


if __name__ == '__main__':
    main()
