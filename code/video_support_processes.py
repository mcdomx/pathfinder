# Pathfinder - Support Script - video_support_processes.py
# Author - Mark McDonald (2019)

# This module contains various support methods to create
# video files from MAT files and JPG files.

import os
import scipy.io
import numpy as np
import cv2
import math
from keras.preprocessing.image import ImageDataGenerator


def create_superimposed_video_from_MATFiles(source_dir, target_dir):
    """
    Creates an MP4 video file with mask overlay from directory of MAT files.

    :param source_dir: directory containing *.mat files
    :param target_dir: directory where the video will be saved
    :return: List of filepaths for each video created
    """

    video_paths = []
    for item in os.listdir(source_dir):
        if item[0] == '.': continue
        if os.path.isdir(os.path.join(source_dir, item)):
            scene_name = item[-4:]
            scene_dir = os.path.join(source_dir, item)

            # Loop MAT files in supplied directory and extract images and masks
            images = []
            masks = []

            image_paths = sorted(os.listdir(scene_dir))

            for item in image_paths:

                # skip hidden and non-mat files
                if item[0] == '.' or item[-3:] != 'mat': continue

                # get the mat file data
                try:
                    mat_file = scipy.io.loadmat(os.path.join(scene_dir, item))
                except:
                    print("Can't load mat file:", os.path.join(scene_dir, item))

                try:
                    image = mat_file['im_rgb']
                except:
                    print("Missing 'im_rbg' element: ", os.path.join(scene_dir, item))

                try:
                    mask = mat_file['manual_human_labeling_mask']
                except:
                    print("Missing 'manual_human_labeling_mask' element: ",
                          os.path.join(scene_dir, item))

                # rotate images so they are horizontal
                image = np.moveaxis(image, 0, 1)
                mask = np.moveaxis(mask, 0, 1)

                images.append(image)
                masks.append(mask)

            # iterate through all the images and create superimposed images with mask
            superimposed_images = []
            for i, image in enumerate(images):

                # make a displayable mask
                mask = np.zeros((3, 480, 640))
                mask[0] = np.array(masks[i] == 1) * 255
                mask[1] = np.array(masks[i] == 0) * 200
                mask[2] = np.array(masks[i] == 2) * 127
                mask = np.moveaxis(mask, 0, -1)

                # superimpose mask on image
                superimposed_image = np.uint8(image * .6 + mask * .4)

                # Create text overlay for the frame
                # 'putText' function requires item to be saved as file
                overlay_text = "Scene: " + scene_name + " (manual overlay)"
                # need to create jpg file first
                cv2.imwrite("tempfile.jpg", superimposed_image)
                tempfile = cv2.imread("tempfile.jpg")  # read newly created file

                # put overlay
                cv2.putText(tempfile,
                            overlay_text,
                            (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .75,
                            (255, 255, 255),
                            2,
                            lineType=cv2.LINE_AA)

                superimposed_images.append(tempfile)

            try:
                os.remove("tempfile.jpg")
            except:
                print("Could not remove tempfile.jpg")

            # Create video of superimposed images
            video_paths.append(create_video_file(superimposed_images,
                                                 target_dir,
                                                 scene_name))

        else:
            print("Directory supplied is expected to contain subdirectories of scenes.")
            print("Skipped: ".format(source_dir + "/" + item))

    # Outermost for loop is complete.  Return paths to videos created.
    return video_paths


def create_video_file(image_list, target_dir, file_name):
    """
    Creates an MP4 video file based on list of arrays.

    :param image_list: list of 3-channel np.array images that make up video
    :param target_dir: directory where the video will be saved
    :param file_name: file name of video file to be saved
    :return: Path to video created
    """

    # Make sure file is appended with filetype
    if file_name[-4:] != ".mp4": file_name = file_name + ".mp4"

    video_path = os.path.join(target_dir, file_name)

    # Determine the width and height from the first image
    height, width, channels = image_list[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

    for image in image_list:
        out.write(image)  # Write out frame to video

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    video_path = os.path.join(target_dir, file_name)

    print("Video created {}".format(video_path))

    return video_path


def create_video_with_mask(source_dir,
                           model,
                           description,
                           target_dir,
                           filename=None,
                           batch_size=None,
                           overlay_text=None,
                           color=None,
                           fps=20.0,
                           follow_bias=0.9,
                           follow_intensity_adj=1.75,
                           avoid_bias=.7,
                           avoid_intensity_adj=.5,
                           img_mask_ratio=.5):

    """
    Creates an MP4 video file with mask overlay generated from model.

    :param source_dir: directory containing a 'data' directory that contains base images
    :param model: Keras trained model that will generate mask based on images
    :param description: dictionary containing entries describing model
    :param target_dir: directory where video will be saved
    :param filename: (opt) filename to save video as. Defaults to overlay_text if not provided
    :param batch_size: Number of images to convert at a time.  If none, all files in dir will be converted at once
    :param overlay_text: text that will be overlaied on top line of video
    :param color: (bool) If true, video will be in color.  If empty, defaults to description value.
    :param fps: Frames per second. Default 20 fps.
    :param follow_bias: Must be >0.  >1 increases follow area.  <1 decreases follow area.
    :param follow_intensity_adj: Must be >0.  >1 increases intensity of color.
    :param avoid_bias: Must be >0.  >1 increases avoid area.  <1 decreases avoid area.
    :param avoid_intensity_adj: Must be >0.  >1 increases intensity of color.
    :param img_mask_ratio: Must be >0 and <1.  Percentage of image intensity under mask areas.
    :return: Path to video created
    """
    output_height = description["output_height"]
    output_width = description["output_width"]

    if overlay_text is None:
        overlay_text = description["name"]

    if filename is None:
        filename = overlay_text

    # if color argument is not supplied, use color from description dictionary
    # defaults to 'rgb', color=True
    if color is None:
        color = description["color"]

    # count the number of valid images in source directory
    num_images = 0
    for img_file in os.listdir(os.path.join(source_dir, 'images', 'data')):
        if img_file[0] != '.' and img_file[-3:] == "jpg":
            num_images += 1
    if num_images == 0:
        print("No images to process in dir: ", source_dir)
        return

    if batch_size is None:
        batch_size = num_images

    # Retrieve generator for test images and get batch of all images
    generator = get_test_img_generator(os.path.join(source_dir, 'images'),
                                       description,
                                       batch_size=batch_size,
                                       color=color)

    # steps is the number of times a batch needs to be called to get all images
    steps = math.ceil(num_images / batch_size)

    print("Processing {} images in {} batches.".format(num_images, steps))

    # Make sure file is appended with filetype
    if filename[-4:] != ".mp4": filename = filename + ".mp4"

    video_path = os.path.join(target_dir, filename)
    print("Creating video name: ", video_path)
    print("FPS: ", fps)

    # Define the codec and open VideoWriter stream
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (output_width, output_height))

    superimposed_images = []
    zero_length = False
    # for each batch, create superimposed images and write to video file
    for step in range(steps):

        print("Processing batch {}/{}".format(step+1, steps))

        images = next(generator)

        # skip iteration if no images
        # if you skipped a batch with no images already, break
        if len(images) == 0:
            if zero_length is True: break
            zero_length = True
            continue

        # generate masks from images using model
        masks = np.uint8(model.predict(np.asarray(images)))

        # creates 3-channel masks with follow and avoid heatmaps in RGB
        new_masks = create_follow_and_avoid_mask_layers(masks,
                                                        follow_bias=follow_bias,
                                                        follow_intensity_adj=follow_intensity_adj,
                                                        avoid_bias=avoid_bias,
                                                        avoid_intensity_adj=avoid_intensity_adj)

        # Keras image generator generates in RGB format
        # Convert to GBR to work with OpenCV
        new_masks = new_masks[..., ::-1]
        images = images[..., ::-1]

        # generate superimposed images
        # opencv saves and reads in BGR
        for i, mask in enumerate(new_masks):
            # Knockout portions of image that include activated overlay values
            merged = mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]
            adj_array = np.array(merged != 0) * img_mask_ratio + (np.array(merged == 0))
            combined = np.zeros((images[i].shape[0], images[i].shape[1], images[i].shape[2]))
            combined[:, :, 0] = images[i][:, :, 0] * adj_array
            combined[:, :, 1] = images[i][:, :, 1] * adj_array
            combined[:, :, 2] = images[i][:, :, 2] * adj_array

            superimposed_image = np.uint8(combined + mask * (1 - img_mask_ratio))

            # Create text overlay for the frame
            # need to create jpg file first
            cv2.imwrite("tempfile.jpg", superimposed_image)
            tempfile = cv2.imread("tempfile.jpg")  # read newly created file
            cv2.putText(tempfile,
                        overlay_text,
                        (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .30,
                        (255, 255, 255),
                        1,
                        lineType=cv2.LINE_AA)

            superimposed_images.append(tempfile)

    for image in superimposed_images:
        out.write(image)  # Write out frame to video

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    try:
        os.remove("tempfile.jpg")
    except:
        print("Could not remove tempfile.jpg")

    print("Video created {}".format(video_path))


def create_follow_and_avoid_mask_layers(masks,
                                        follow_bias=0.9, follow_intensity_adj=1.75,
                                        avoid_bias=.7, avoid_intensity_adj=.5):
    """
    Accepts 1-channel masks and returns 3-channel masks with heatmaps

    :param masks: Batch of 1-channel masks
    :param follow_bias: If 0, no follow mask created.  >1 increases follow area.  <1 decreases follow area.
    :param follow_intensity_adj: Must be >0.  >1 increases intensity of color.
    :param avoid_bias: If 0, no avoid mask created.  >1 increases avoid area.  <1 decreases avoid area.
    :param avoid_intensity_adj: Must be >0.  >1 increases intensity of color.
    :return: batch of 3-channel masks with follow and avoid heatmaps
    """

    # rescale from 0 to 255
    adj_masks = masks - np.amin(masks)  # set smallest value to zero
    scale = 255 / np.amax(adj_masks)  # determine scale so max value will be 255
    adj_masks = np.uint8(adj_masks * scale)
    avg = np.uint8(np.average(adj_masks))

    # Heatmap layers will be added to new 3-channel mask
    rgb_mask = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3))

    # Create follow heatmap
    if follow_bias:
        follow_layer = np.array(adj_masks > (avg / follow_bias)) * adj_masks
        # Adjust color intensity
        follow_intensed_layer = follow_layer * follow_intensity_adj
        # Prevent clipping
        follow_layer = np.uint8(
            np.array(follow_intensed_layer >= 255) * 255 + np.array(follow_intensed_layer < 255) * follow_intensed_layer)
        # add follow channel to new mask
        rgb_mask[:, :, :, 1] = follow_layer[:, :, :, 0]

    # Create avoid heatmap
    if avoid_bias:
        avoid_layer = 0 - (np.array(adj_masks < (avg * avoid_bias)) * adj_masks)
        # Adjust color intensity
        avoid_intensed_layer = avoid_layer * avoid_intensity_adj
        # Prevent clipping
        avoid_layer = np.uint8(
            np.array(avoid_intensed_layer >= 255) * 255 + np.array(avoid_intensed_layer < 255) * avoid_intensed_layer)
        # knockout portions of avoid channel that intersect with follow channel
        if follow_bias:
            avoid_layer = np.uint8((1 - np.array(follow_layer / 255)) * avoid_layer)
        # add avoid channel to new mask
        rgb_mask[:, :, :, 0] = avoid_layer[:, :, :, 0]

    return np.uint8(rgb_mask)


def get_test_img_generator(source_dir, description, batch_size=None, color=None):
    """
    Defines an unshuffled image generator for files in a source directory.

    :param source_dir: directory containing a 'data' directory that contains base images
    :param description: dictionary containing entries describing model that will use generator
    :param batch_size: (opt) Batch size for each generator call. If None, uses description value.
    :param color: (opt)(bool) Color or black and white. If None, uses description value.
    :return: Configured Keras ImageDataGenerator() object.
    """

    print("Creating test image generator:")
    print("\t"+source_dir)
    input_height = description["input_height"]
    input_width = description["input_width"]

    # if color argument is not supplied, use color from description dictionary
    # defaults to 'rgb', color=True
    color_mode = 'rgb'
    if color is None:
        color = description["color"]
        if not color:
            color_mode = 'grayscale'

    if batch_size is None:
        batch_size = description["batch_size"]

    # Configure image generator
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        source_dir,
        target_size=(input_height, input_width),
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    return image_generator

