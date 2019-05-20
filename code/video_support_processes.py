# Pathfinder - Support Script - video_support_processes.py
# Author - Mark McDonald (2019)

# This module contains various support methods to create
# video files from frames.

import os
import scipy.io
import numpy as np
import cv2
import math
from keras.preprocessing.image import ImageDataGenerator


# Creates an MP4 video file with mask overlay
# Arguments:
#   source_dir: directory containing *.mat files
#   target_dir: directory where the video will be saved
# Return Value: List of path for each video created.
# Summary:
#   Image and mask data is extracted from each *.mat file.
#   Each mask is overlayed on its respective image.
#   Each overlayed image is added to a video sequence and saved
def create_superimposed_video_from_MATFiles(source_dir, target_dir):

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

# Creates an MP4 video file based on list of arrays
# Arguments:
#   image_list: list of 3-channel np.array images that make up video
#   target_dir: directory where the video will be saved
#   file_name: name of video file to be saved
# Return Value: Path to video created.
# Summary:
#   Uses cv2 to stitch images together into video file
def create_video_file(image_list, target_dir, file_name):

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


# Creates an MP4 overlay video file on mask created from model
# Arguments:
#   source_dir: directory containing a 'data' directory that contains base images
#   model: Keras trained model that will generate mask based on images
#   description: dictionary containing entries describing model
#   target_dir: directory where video will be saved
#   batch_size: Number of images to convert at a time.  If none, all files in dir will be converted at once
#   overlay_text: text that will be overlayed on video
#   color(boolean): whether or not the test data will be color
#   bias(float): value between -10 and +10.  More(+) of less(-) aggressive with masked areas.
# Return Value: Path to video created
# The video will be saved with a name containing relevant model parameters.
# Requires method which returns a configured generator for test images.
# Summary:
#   Retrieves test images from generator.
#   Creates a mask for each image and creates overlay image
#   Calls function to create video based on list of np.array images.
def create_video_with_test_data(source_dir,
                                model,
                                description,
                                target_dir,
                                filename=None,
                                batch_size=None,
                                overlay_text=None,
                                color=None,
                                bias=0,
                                fps=20.0,
                                invert=False):

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
    steps = math.ceil(num_images/batch_size)

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


        images = next(generator)

        # skip iteration if no images
        # if you skipped a batch with no images already, break
        if len(images) == 0:
            if zero_length is True: break
            zero_length = True
            continue

        # generate masks from images using model model
        masks = model.predict(np.asarray(images))

        avg = np.average(masks)
        print("Processing batch: {}/{}".format(step + 1, steps), end=" | ")
        print("Max/Min/Avg mask value: {}/{}/{}".format(np.round(np.amax(masks), decimals=2),
                                                        np.round(np.amin(masks), decimals=2),
                                                        np.round(avg, decimals=2)))

        # A bias will move the average to show more or less highlighted areas
        if bias < 0:
            bias = (1 - avg) * (bias / 10)
            avg += bias
        if bias > 0:
            bias = avg * (bias / 10)
            avg -= bias

        follow_layer = 1
        avoid_layer = 0
        if invert:
            follow_layer = 0
            avoid_layer = 1

        # generate superimposed images
        for i, mask in enumerate(masks):

            # mask is created by highlighting pixels that exceed the average
            mask = cv2.resize(mask, (output_width, output_height))
            new_mask = np.zeros((3, mask.shape[0], mask.shape[1]))
            new_mask[avoid_layer] = np.array(mask <= avg) * 200  # Red Layer
            new_mask[follow_layer] = np.array(mask >= avg) * 200  # Green Layer
            new_mask = np.moveaxis(new_mask, 0, 2)
            new_mask = np.uint8(new_mask)

            superimposed_image = np.uint8(images[i]*.7 + new_mask*.3)

            # Create text overlay for the frame
            # need to create jpg file first
            cv2.imwrite("tempfile.jpg", superimposed_image)
            tempfile = cv2.imread("tempfile.jpg")  # read newly created file
            cv2.putText(tempfile,
                        overlay_text,
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .40,
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


# Define a test image generator for all files in a directory as a single batch
# Arguments:
#   source_dir: directory containing a 'data' directory that contains base images
#   description: dictionary containing entries describing model that will use generator
#   color(boolean): Whether or not yielded images are color or black and white.
#                   None (missing color argument) will force use of color
#                   variable from description dictionary.
# Return Value: Configured ImageGenerator that can be called with next(<generator_name>)
# The video will be saved with a name containing relevant model parameters.
# Requires method which returns a configured generator for test images.
# Summary:
#   Retrieves test images from generator.
#   Creates a mask for each image and creates overlay image
#   Calls function to create video based on list of np.array images.
def get_test_img_generator(source_dir, description, batch_size=None, color=None):

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

