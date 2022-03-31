import numpy as np
import cv2
from PIL import Image
import scipy
import cv2
from moviepy.editor import VideoFileClip
# from IPython.display import HTML
from keras.models import load_model


# Load Keras model
model = load_model('Model/LaneNet.h5')


# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    # Get image ready for feeding into model
    small_img = cv2.resize(image, (160, 80))
    # small_img = np.array(Image.fromarray(image).resize((3,(80, 160))))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = cv2.resize(lane_drawn, (1280, 720))
    image = cv2.resize(image, (1280, 720))
    # lane_image = np.array(Image.fromarray(lane_drawn).resize((3,(1080,1920))))
    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0, dtype = cv2.CV_8UC3)
    return result


# 运行车道线检测程序
lanes = Lanes()

# Location of the input video
clip1 = VideoFileClip("video/4.MP4")

# Where to save the output video
vid_output = 'video/out_4.mp4'

vid_clip = clip1.fl_image(road_lines)
vid_clip.write_videofile(vid_output, audio = False)
