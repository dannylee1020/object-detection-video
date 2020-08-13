import streamlit as st
import numpy as np
import imutils
import time
import cv2
import os

import config
import time



def detect_objects(video, confidence_threshold, nms_threshold):

# get video frames and pass to YOLO for output

	# load YOLO from cv2.dnn 
	# determine only the output layer names we need from YOLO
	net = cv2.dnn.readNetFromDarknet(config.CONFIG_PATH, config.MODEL_PATH)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# initialize video stream, pointer to output video file and grabbing frame dimension
	vs = cv2.VideoCapture(video)
	fps = vs.get(cv2.CAP_PROP_FPS)
	writer_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
	writer_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

	writer = None
	(W,H) = (None, None)

	# determine the total number of frames in a video
	try:
		prop = cv2.CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
		total = int(vs.get(prop))
		print(f"[INFO] {total} frames in the video")

	# if error occurs print
	except:
		print(f"[INFO] {total} frames in the video")
		total = -1

	# loop over on entire video frames
	while True:
		# read next frame
		(grabbed, frame) = vs.read()

		# if no frame is grabbed, we reached the end of video, so break the loop
		if not grabbed:
			break
		# if the frame dimensions are empty, grab them
		if W is None or H is None:
			(H,W) = frame.shape[:2]

		# build blob and feed forward to YOLO to get bounding boxes and probability
		blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB = True, crop = False)
		start = time.time()
		net.setInput(blob)
		layerOutputs = net.forward(ln)
		end = time.time()

 
# get metrics from YOLO

		boxes = []
		confidences = []
		classIDs = []

		# loop over each output from layeroutputs
		for output in layerOutputs:
			# loop over each detecton in output
			for detection in output:
				# extract score, ids and confidence of current object detection
				score = detection[5:]
				classID = np.argmax(score)
				confidence = score[classID]

				# filter out weak detections with confidence threshold
				if confidence > confidence_threshold:
					# scale bounding box coordinates back relative to image size
					# YOLO spits out center (x,y) of bounding boxes followed by 
					# boxes width and heigth
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype('int')

					# grab top left coordinate of the box
					x = int(centerX - (width/2))
					y = int(centerY - (height/2))

					boxes.append([x,y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)


# Apply Non-Max Suppression, draw boxes and write output video 

		idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
		# ensure detection exists
		if len(idxs) > 0:
			for i in idxs.flatten():
				# getting box coordinates
				(x,y) = (boxes[i][0], boxes[i][1])
				(w,h) = (boxes[i][2], boxes[i][3])

				# color and draw boxes
				color = [int(c) for c in config.COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
				text = f"{config.LABELS[classIDs[i]]}: {confidences[i]}"
				cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


		if writer is None:
			# initialize video writer by setting fourcc
			# and writing output video to output path
			# fourcc = cv2.VideoWriter_fourcc(*'H264')
			fourcc = cv2.VideoWriter_fourcc(*'H264')
			writer = cv2.VideoWriter(config.OUTPUT_PATH, fourcc, fps, (writer_width, writer_height), True)

			if total > 0:
				elap = (end - start)
				print(f"[INFO] single frame took {round(elap/60,2)} minutes")
				print(f"[INFO] total estimated time to finish: {(elap*total)/60} minutes")


		
		writer.write(frame)

	writer.release()
	vs.release()

	return total, elap


