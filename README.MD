# Object Detection in Video Stream with YOLO V3

## Project Overview
Detecting objects in a video using YOLO V3 algorithm. The approach is quite similar to detecting images with YOLO. We get every frame of a video like an image and detect objects at that frame using yolo. Then draw the boxes, labels and iterate through all the frame in a given video. Adjust the confidence and nms threshold to see how the algorithm's detections change. The annotated video will be stored in the output folder in .mp4 file format. Make sure to add yolov3.weights file to the model folder to build and run with docker. 

## Run with Docker
from the root directory, run

		docker build -t dannylee1020/object_detection_video .
		docker run -p 8501:8501 dannylee1020/object_detection_video:latest

then visit [localhost:8501](https://localhost:8501) for streamlit app

## Reference
[YOLO](https://pjreddie.com/darknet/yolo/)
<br>
[Object Detection with OpenCV](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
<br>
[Streamlit](https://www.streamlit.io/)