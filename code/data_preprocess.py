# Pathfinder - Support Script - data_preprocesses.py
# Author - Mark McDonald (2019)

# This module contains various support methods to create
# files and directories needed to execute the pathfinder
# project.

import os
import scipy.io
import imageio
import numpy as np
import random
import shutil
from keras.preprocessing import image as kimage
import cv2


# Sets a series of directories for training and validation data
# Arguments:
#   base_directory: directory under which supporting directories are created
#   remove_existing_structure_bool: If True, removes previously created
#                                   project directories. (except 'results' directory)
# Return Value: train_data_dir, train_data_video_dir, val_data_dir, test_data_dir, results_dir
# Summary:
#   Project requires specific directories for train, validation, results and test data.
#   These will be created with this method.
#   Note that image and mask directories have 'data' subdirectory which is
#   necessary for data generators.
# Requires a full path to the base directory under which
# supporting directories will be created.
def set_directories(base_directory, remove_existing_structure_bool):

    if not os.path.isdir(base_directory):
        print("ERROR: You must first create a base directory.")
        print("ERROR: Create directory and try again: ", base_directory)
        print("No directories were removed or created.")
        exit(0)

    # Define directory paths
    train_data_dir = os.path.join(base_directory, 'train')
    train_data_images_dir = os.path.join(train_data_dir, 'images')
    train_data_labels_dir = os.path.join(train_data_dir, 'masks')
    train_data_video_dir = os.path.join(train_data_dir, 'video')

    val_data_dir = os.path.join(base_directory, 'val')
    val_data_images_dir = os.path.join(val_data_dir, 'images')
    val_data_labels_dir = os.path.join(val_data_dir, 'masks')

    test_data_dir = os.path.join(base_directory, 'test')
    test_data_images_dir = os.path.join(test_data_dir, 'images')

    results_dir = os.path.join(base_directory, 'results')

    # Create Train Directories
    # Note: data directory is needed for use of data generators
    try:
        os.mkdir(train_data_dir)
        print("Created Train Directory: {}".format(train_data_dir))
    except:
        if remove_existing_structure_bool:
            shutil.rmtree(train_data_dir)
            os.mkdir(train_data_dir)
            print("Erased and Created Train Directory: {}".format(train_data_dir))
        else:
            print("Train Directory already existed and was retained: {}".format(train_data_dir))

    try:
        os.mkdir(train_data_images_dir)
        print("Created Train Images Directory: {}".format(train_data_images_dir))
    except:
        print("Train Images directory already exists: {}".format(train_data_images_dir))

    try:
        os.mkdir(os.path.join(train_data_images_dir, 'data'))
        print("Created Train Images Data Directory: {}".format(os.path.join(train_data_images_dir, 'data')))
    except:
        print("Train Images data directory already exists: {}".format(os.path.join(train_data_images_dir, 'data')))

    try:
        os.mkdir(train_data_labels_dir)
        print("Created Train Masks Directory: {}".format(train_data_labels_dir))
    except:
        print("Train Masks directory already exists: {}".format(train_data_labels_dir))

    try:
        os.mkdir(os.path.join(train_data_labels_dir, 'data'))
        print("Created Train Masks Data Directory: {}".format(os.path.join(train_data_labels_dir, 'data')))
    except:
        print("Train Masks data directory already exists: {}".format(os.path.join(train_data_labels_dir, 'data')))

    try:
        os.mkdir(train_data_video_dir)
        print("Created Video Dir: {}".format(train_data_video_dir))
    except:
        print("Video Dir already exists: {}".format(train_data_video_dir))

    # Create Validation Directories
    try:
        os.mkdir(val_data_dir)
        print("Created Validation Directory: {}".format(val_data_dir))
    except:
        if remove_existing_structure_bool:
            shutil.rmtree(val_data_dir)
            os.mkdir(val_data_dir)
            print("Erased and Created Validation Directory: {}".format(val_data_dir))
        else:
            print("Validation Directory already existed and was retained: {}".format(val_data_dir))

    try:
        os.mkdir(val_data_images_dir)
        print("Created Validation Images Dir: {}".format(val_data_images_dir))
    except:
        print("Validation Images Dir already exists: {}".format(val_data_images_dir))

    try:
        os.mkdir(os.path.join(val_data_images_dir, 'data'))
        print("Created Validation Data Images Dir: {}".format(os.path.join(val_data_images_dir, 'data')))
    except:
        print("Validation Data Images Dir already exists: {}".format(os.path.join(val_data_images_dir, 'data')))

    try:
        os.mkdir(val_data_labels_dir)
        print("Created Validation Masks Dir: {}".format(val_data_labels_dir))
    except:
        print("Validation Masks Dir already exists: {}".format(val_data_labels_dir))

    try:
        os.mkdir(os.path.join(val_data_labels_dir, 'data'))
        print("Created Validation Data Masks Dir: {}".format(os.path.join(val_data_labels_dir, 'data')))
    except:
        print("Validation Data Masks Dir already exists: {}".format(os.path.join(val_data_labels_dir, 'data')))

    # Create Test Directories
    try:
        os.mkdir(test_data_dir)
        print("Created Test Directory: {}".format(test_data_dir))
    except:
        if remove_existing_structure_bool:
            shutil.rmtree(test_data_dir)
            os.mkdir(test_data_dir)
            print("Erased and Created Validation Directory: {}".format(test_data_dir))
        else:
            print("Validation Directory already existed and was retained: {}".format(test_data_dir))

    try:
        os.mkdir(test_data_images_dir)
        print("Created Test Images Dir: {}".format(test_data_images_dir))
    except:
        print("Test Images Dir already exists: {}".format(test_data_images_dir))

    try:
        os.mkdir(os.path.join(test_data_images_dir, 'data'))
        print("Created Test Data Images Dir: {}".format(os.path.join(test_data_images_dir, 'data')))
    except:
        print("Test Data Images Dir already exists: {}".format(os.path.join(test_data_images_dir, 'data')))

    # Create Results Directory
    # If exists, 'results' directory is not recreated
    try:
        os.mkdir(results_dir)
        print("Created Results Directory: {}".format(results_dir))
    except:
        print("Results Directory already existed and was retained: {}".format(results_dir))

    return train_data_dir, train_data_video_dir, val_data_dir, test_data_dir, results_dir


# will extract data from a single mat file and save the image and mask data
# if a mask does not exist, no mask actions will be taken
# Arguments:
#   mat_file_path:  full path the the mat file
#   img_dir:        full path to the directory where the image should be saved
#   mask_dir:       full path to the directory where the mask will be saved
#   save_to_name:   a name that will be used for both the image and mask file
def extract_matfile_data(mat_file_path, img_dir, mask_dir, save_to_name):
    # get the mat file data
    try:
        mat_file = scipy.io.loadmat(mat_file_path)
    except:
        print("Can't load mat file: ", mat_file_path)

    # retrieve the frame's array data
    try:
        frame = mat_file['im_rgb']
        img_exists = True
    except:
        img_exists = False

    # retrieve the labeled mask array data
    try:
        mask = mat_file['manual_human_labeling_mask']
        mask_exists = True
    except:
        mask_exists = False

    # rotate images so they are horizontal
    frame = np.moveaxis(frame, 0, 1)

    if mask_exists:
        mask = np.moveaxis(mask, 0, 1)

        # need to add this dimension to mask so when using data generator, it sees it as gray scale
        mask = np.expand_dims(mask, axis=2)

        # convert mask to JPEG image and save to disk
        imageio.imwrite(os.path.join(mask_dir, str(save_to_name) + ".jpg"), mask, format='JPEG-PIL')

    # convert image to JPEG image and save to disk
    imageio.imwrite(os.path.join(img_dir, str(save_to_name) + ".jpg"), frame, format='JPEG-PIL')

    return img_exists, mask_exists


# After supporting directories are created, this method will extract
# image and mask data from mat files and place them in respective
# project directories.
# Arguments:
#   orig_data_path: a path that includes subdirectories that contain
#                   *.mat files.  The subdirectories under orig_data_path
#                   should represent a scene and the mat files under each scene
#                   directory represent the frames in the video sequence.
#   target_dir: An existing directory where 'images/data' and 'masks/data'
#               subdirectories already exist.  Extracted image and mask data will be saved
#               in these subdirectories.  The frames from all scenes will be extracted
#               to the same directory.
# Return Value: train_data_dir, train_data_video_dir, val_data_dir, test_data_dir, results_dir
# Note:
#   MAT files without a mask will ignore mask actions.
def create_img_and_mask_data(orig_data_path, target_dir):
    if not os.path.isdir(orig_data_path):
        print("Provided path must be a directory.")
        return

    target_img_dir = os.path.join(target_dir, 'images', 'data')
    target_mask_dir = os.path.join(target_dir, 'masks', 'data')

    mask_exists = False
    img_exists = False

    # Get the scene name from the directory
    # Scene names start with DS and are 4 characters long
    path_split = orig_data_path.split('_')
    for item in path_split:
        if item[0:2] == "DS" and len(item)==4:
            scene_name = item
            break

    # First get all MAT files and then extract data into training directories
    for item in os.listdir(orig_data_path):
        if item[0] == '.': continue
        new_path = os.path.join(orig_data_path, item)

        # if the item is a directory, recurse
        if os.path.isdir(new_path):
            create_img_and_mask_data(new_path, target_dir)
            continue

        # only process mat files
        elif item[-4:] != ".mat":
            continue

        # if the item is a file extract_and_save_train_data
        else:
            filename = scene_name + '_' + item[0:len(item) - 4]
            img_exists, mask_exists = extract_matfile_data(new_path, target_img_dir, target_mask_dir, str(filename))

    if len(orig_data_path) > 0:
        if img_exists or mask_exists:
            print("\nSource Data Directory: ", orig_data_path)
        if img_exists:
            print("Extracted Images: ", target_img_dir)
        if mask_exists:
            print("Extracted Masks: ", target_mask_dir)


# Move the validation set to validation directories from training set
# After supporting directories are created, this method will extract
# image and mask data from mat files and place them in respective
# project directories.
# Arguments:
#   source_dir: a path that includes image and mask subdirectories from
#               which,  a portion of image files and respective mask files
#               will be moved to a validation directory.
#   target_dir: An existing directory where selected images and masks will
#               be moved to.
#   val_split:  A percentage of items from teh source_dir that will be moved to the
#               target_dir.
# Return Value: none
def create_val_set(source_dir, target_dir, val_split):
    source_img_dir = os.path.join(source_dir, 'images', 'data')
    source_mask_dir = os.path.join(source_dir, 'masks', 'data')
    target_img_dir = os.path.join(target_dir, 'images', 'data')
    target_mask_dir = os.path.join(target_dir, 'masks', 'data')

    # This will hold a random series of indicies that will be moved to the target
    move_indices = []
    files_list = [x for x in sorted(os.listdir(source_img_dir)) if not x.startswith(".")]
    masks_list = [x for x in sorted(os.listdir(source_mask_dir)) if not x.startswith(".")]
    all_indicies = list(range(0, len(files_list) - 1))
    num_images_to_move = int(len(files_list) * val_split)

    move_indices = random.sample(all_indicies, num_images_to_move)

    # move the images and masks with indicies in the test_indicies list to the train directory
    move_indices = sorted(move_indices, reverse=True)
    for i in move_indices:
        shutil.move(os.path.join(source_img_dir, files_list[i]), os.path.join(target_img_dir, files_list[i]))
        shutil.move(os.path.join(source_mask_dir, masks_list[i]), os.path.join(target_mask_dir, masks_list[i]))

    print("\nMoved {} validation images to {}".format(num_images_to_move, target_img_dir))
    print("Moved {} validation images to {}".format(num_images_to_move, target_mask_dir))


# Recursively iterates over subdirectories to find original masks and convert to
# binary masks.
# Arguments:
#   jpeg_mask_filepath: a directory or file that contains or is a mask
# Return Value: 1 if file converted
# Note:
#       Only masks that have a min value of 0 and max value of 2 will be converted.
def convert_jpg_mask_to_binary_mask(jpeg_mask_filepath):
    # If filepath is a directory iterate and process all files
    converted = 0
    if os.path.isdir(jpeg_mask_filepath):
        for item in os.listdir(jpeg_mask_filepath):
            if item[0] == '.': continue
            item = os.path.join(jpeg_mask_filepath, item)
            converted += convert_jpg_mask_to_binary_mask(item)  # recurse

    # if the filepath is a file, do the conversion
    else:
        # load current mask
        try:
            mask = kimage.load_img(jpeg_mask_filepath)
        except:
            print("Could not load filepath: " + jpeg_mask_filepath)
            return 0

        # Test that we are converting an original mask
        # Check that elements of mask are between 0 and 2
        if np.amin(mask) == 0 and np.amax(mask) == 2:
            print("Not an original mask. Didn't convert: {}".format(jpeg_mask_filepath))
            return 0

        # convert image to an array
        mask = np.array(mask)

        # move the channels to the first dimension
        mask = np.moveaxis(mask, 2, 0)

        # change the 0's to 1's and everything else 0
        binary_mask = np.array(mask[0] == 0)
        binary_mask = np.array(binary_mask, dtype='uint8')

        # Write over the original mask with the new binary mask
        # imageio.imwrite(jpeg_mask_filepath, binary_mask, format='JPEG-PIL')
        return cv2.imwrite(jpeg_mask_filepath, binary_mask)

    print("Converted {} masks in {}".format(converted, jpeg_mask_filepath))
    return 0




