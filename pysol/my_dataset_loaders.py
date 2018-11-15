import os
from os import path
import numpy
import cv2
import math
import sys
import scipy.io

def show_progress(max_val, present_val):
    progress = present_val / (max_val+1) * 100
    sys.stdout.write("Progress: %d%%   \r" % (progress) )
    sys.stdout.flush()

def load_train_dataset():
    train_dir = '/home/himanshu/Documents/Projects/data/mymulticlass/train'
    categories = os.listdir(train_dir)
    max_sample_per_category = []            # maximum samples in each category
    sample_counter_per_category = []        # samples processed in each category
    files_per_category = []                 # all files per category

    # Count all samples
    for i in range(len(categories)):
        t_files = os.listdir(path.join(train_dir, categories[i]))       # read all files
        files_per_category.append(t_files.copy())                       # append list of files
        max_sample_per_category.append(len(t_files))                    # append file count
        sample_counter_per_category.append(0)                           # initialize with zero
        print('# ', categories[i], ' = ', len(t_files))

    X_train = numpy.zeros( (sum(max_sample_per_category)*2,144,144,3), dtype=numpy.uint8 )
    y_train = numpy.zeros( sum(max_sample_per_category)*2, dtype=numpy.uint8 )

    # Load data without flips
    print('Loading train data without flips...')
    for i in range(max(max_sample_per_category)):     # for maximum number of samples in a train category
        show_progress(max_val=max(max_sample_per_category), present_val=max(sample_counter_per_category))
        for j in range(len(categories)):
            if sample_counter_per_category[j] < max_sample_per_category[j]:       # if all samples from that category are not read
                im_path = path.join(train_dir, categories[j], files_per_category[j][i])
                im1 = cv2.imread(im_path)
                assert im1.shape[2] == 3    # 3 channel assertion
                X_train[sum(sample_counter_per_category),:,:,:] = cv2.resize(im1, dsize=(144, 144), interpolation=cv2.INTER_CUBIC)      # resize image
                y_train[sum(sample_counter_per_category)] = j+1   # assign category
                sample_counter_per_category[j] = sample_counter_per_category[j] + 1

    counter_offset = sum(sample_counter_per_category)
    # Reset sample counter
    for i in range(len(categories)):
        sample_counter_per_category[i] = 0                       # initialize with zero

    # Load data with flips
    print('Loading train data with flips...')
    for i in range(max(max_sample_per_category)):     # for maximum number of samples in a train category
        show_progress(max_val=max(max_sample_per_category), present_val=max(sample_counter_per_category))
        for j in range(len(categories)):
            if sample_counter_per_category[j] < max_sample_per_category[j]:       # if all samples from that category are not read
                im_path = path.join(train_dir, categories[j], files_per_category[j][i])
                im1 = cv2.imread(im_path)
                assert im1.shape[2] == 3    # 3 channel assertion
                im2 = cv2.flip( im1, 1 )    # horizontal flip
                X_train[counter_offset+sum(sample_counter_per_category),:,:,:] = cv2.resize(im2, dsize=(144, 144), interpolation=cv2.INTER_CUBIC)      # resize image
                y_train[counter_offset+sum(sample_counter_per_category)] = j+1   # assign category
                sample_counter_per_category[j] = sample_counter_per_category[j] + 1

    return [X_train, y_train]

def load_test_dataset():
    test_dir = '/home/himanshu/Documents/Projects/data/mymulticlass/test'
    categories = os.listdir(test_dir)
    max_sample_per_category = []            # maximum samples in each category
    sample_counter_per_category = []        # samples processed in each category
    files_per_category = []                 # all files per category

    # Count all samples
    for i in range(len(categories)):
        t_files = os.listdir(path.join(test_dir, categories[i]))       # read all files
        files_per_category.append(t_files.copy())                       # append list of files
        max_sample_per_category.append(len(t_files))                    # append file count
        sample_counter_per_category.append(0)                           # initialize with zero
        print('# ', categories[i], ' = ', len(t_files))

    X_test = numpy.zeros( (sum(max_sample_per_category),144,144,3), dtype=numpy.uint8 )
    y_test = numpy.zeros( sum(max_sample_per_category), dtype=numpy.uint8 )

    # Load data without flips
    print('Loading test data without flips...')
    for i in range(max(max_sample_per_category)):     # for maximum number of samples in a test category
        show_progress(max_val=max(max_sample_per_category), present_val=max(sample_counter_per_category))
        for j in range(len(categories)):
            if sample_counter_per_category[j] < max_sample_per_category[j]:       # if all samples from that category are not read
                im_path = path.join(test_dir, categories[j], files_per_category[j][i])
                im1 = cv2.imread(im_path)
                assert im1.shape[2] == 3    # 3 channel assertion
                X_test[sum(sample_counter_per_category),:,:,:] = cv2.resize(im1, dsize=(144, 144), interpolation=cv2.INTER_CUBIC)      # resize image
                y_test[sum(sample_counter_per_category)] = j+1   # assign category
                sample_counter_per_category[j] = sample_counter_per_category[j] + 1

    return [X_test, y_test]

def load_dataset():
    """ 
    Note: number of samples per category can be different
    mode: train (with augmentation), test (without augmentation)
    """
    [X_train, y_train] = load_train_dataset()
    [X_test, y_test] = load_test_dataset()
    # load images -> flip horizontally -> zero padd
    return [X_train, y_train, X_test, y_test]

def load_dataset_with_UAP(split_th = 0.8, ext='.jpg'):
    """ Default: 80% for training, 20% for testing """
    
    # return [X_train, y_train, X_test, y_test]
    pass

# if __name__ == '__main__':
#     X_train, y_train, X_test, y_test = load_dataset()

#     print ('Dataset loaded.')
#     bool_var = True
#     # while bool_var:
#     #     i = input()
#     i = 10
#     cv2.imshow("Image 1", X_train[i,:,:,:])
#     cv2.imshow("Image 2", X_train[i+1,:,:,:])
#     cv2.imshow("Image 3", X_train[i+2,:,:,:])
#     cv2.imshow("Image 4", X_train[i+3,:,:,:])
#     cv2.imshow("Image 5", X_train[i+4,:,:,:])
#     print(y_train[i], y_train[i+1], y_train[i+2], y_train[i+3], y_train[i+4])
#     # cv2.imshow("Image 1", X_train[int(i),:,:,:])
#     # cv2.imshow("Image 2", X_train[int(i+1),:,:,:])
#     # cv2.imshow("Image 3", X_train[int(i+2),:,:,:])
#     # cv2.imshow("Image 4", X_train[int(i+3),:,:,:])
#     # cv2.imshow("Image 5", X_train[int(i+4),:,:,:])
#     # print(y_train[int(i)], y_train[int(i+1)], y_train[int(i+2)], y_train[int(i+3)], y_train[int(i+4)])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()    # without this jupyter notebook will crash


