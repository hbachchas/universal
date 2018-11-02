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

def load_dataset(split_th = 0.8):
    """ 
    Default: 80% for training, 20% for testing
    Note: number of samples per category can be different
    """
    data_dir = '/home/himanshu/Documents/Projects/data/mymulticlass/category5data'
    categories = os.listdir(data_dir)
    # categories = ['elephant', 'pumpkin', 'tiger', 'tomato', 'watermelon']
    max_sample_per_category = []            # maximum samples in each train category
    sample_counter_per_category = []        # samples processed in each train category
    files_per_category = []                 # all train files per category
    
    # Count all samples
    for i in range(len(categories)):
        t_files = os.listdir(path.join(data_dir, categories[i]))    # read all files
        files_per_category.append(t_files.copy())                   # append list of files
        max_sample_per_category.append(len(t_files))                # append file count
        sample_counter_per_category.append(0)                       # initialize with zero
        print('# ', categories[i], ' = ', len(t_files))

    X = numpy.zeros( (sum(max_sample_per_category)*2,144,144,3), dtype=numpy.uint8 )
    y = numpy.zeros( sum(max_sample_per_category)*2, dtype=numpy.uint8 )

    # Load data without flips
    print('Loading data without flips...')
    for i in range(max(max_sample_per_category)):     # for maximum number of samples in a train category
        show_progress(max_val=max(max_sample_per_category), present_val=max(sample_counter_per_category))
        for j in range(len(categories)):
            if sample_counter_per_category[j] < max_sample_per_category[j]:       # if all samples from that category are not read
                im_path = path.join(data_dir, categories[j], files_per_category[j][i])
                im1 = cv2.imread(im_path)
                assert im1.shape[2] == 3    # 3 channel assertion
                X[sum(sample_counter_per_category),:,:,:] = cv2.resize(im1, dsize=(144, 144), interpolation=cv2.INTER_CUBIC)      # resize image
                y[sum(sample_counter_per_category)] = j+1   # assign category
                sample_counter_per_category[j] = sample_counter_per_category[j] + 1

    counter_offset = sum(sample_counter_per_category)
    # Reset sample counter
    for i in range(len(categories)):
        sample_counter_per_category[i] = 0                       # initialize with zero

    # Load data with flips
    print('Loading data with flips...')
    for i in range(max(max_sample_per_category)):     # for maximum number of samples in a train category
        show_progress(max_val=max(max_sample_per_category), present_val=max(sample_counter_per_category))
        for j in range(len(categories)):
            if sample_counter_per_category[j] < max_sample_per_category[j]:       # if all samples from that category are not read
                im_path = path.join(data_dir, categories[j], files_per_category[j][i])
                im1 = cv2.imread(im_path)
                assert im1.shape[2] == 3    # 3 channel assertion
                im2 = cv2.flip( im1, 1 )    # horizontal flip
                X[counter_offset+sum(sample_counter_per_category),:,:,:] = cv2.resize(im2, dsize=(144, 144), interpolation=cv2.INTER_CUBIC)      # resize image
                y[counter_offset+sum(sample_counter_per_category)] = j+1   # assign category
                sample_counter_per_category[j] = sample_counter_per_category[j] + 1

    training_samples_limit = math.ceil( split_th * counter_offset*2 )
    X_train = X[0:training_samples_limit,:,:,:]
    y_train = y[0:training_samples_limit]
    X_test = X[training_samples_limit:counter_offset*2,:,:,:]
    y_test = y[training_samples_limit:counter_offset*2]
    
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


