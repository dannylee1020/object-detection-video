import numpy as np

## streamlit local build
# DIR_PATH = '/Users/dhyungseoklee/Projects/ML_Pipelines/object-detection-video/'

# Docker Path
DIR_PATH = '/opt/object_detection_video/'

model = 'model/yolov3.weights'
model_config = 'model/yolov3.cfg'
labels = 'model/coco.names'
input_videos = 'videos/'
output_video = 'output/output_video.mp4'

MODEL_PATH = DIR_PATH + model
CONFIG_PATH = DIR_PATH + model_config
LABEL_PATH = DIR_PATH + labels
OUTPUT_PATH = DIR_PATH + output_video
INPUT_PATH = DIR_PATH+ input_videos
VIDEO_PATH = DIR_PATH + input_videos

LABELS = open(LABEL_PATH).read().strip().split('\n')

COLORS = np.random.randint(0, 255, size = (len(LABELS), 3), dtype = 'uint8')

DEFALUT_CONFIDENCE = 0.5
NMS_THRESHOLD = 0.3

