# Import libraries
import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as image
import pathlib
import glob
import imutils
import pickle

#Load model
loaded_model = pickle.load(open('knn_model.pkl', 'rb'))

MODEL = 'yolo\yolov3-face.cfg'
WEIGHT = 'yolo\yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
IMG_WIDTH, IMG_HEIGHT = 416, 416

#function preprocess image
def preprocess(face_image):
    img = cv2.resize(face_image, (300, 300), interpolation = cv2.INTER_AREA)
    img = img / 255.
    img = img.flatten()
    return img


# Set up camera
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT),[0, 0, 0], 1, crop=False)

    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction'
    outs = net.forward(output_layers)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.

    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
    # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1]
            # Extract position data of face area (only area with high confidence)
            if confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                
                # Find the top left point of the bounding box 
                topleft_x = center_x - width/2
                topleft_y = center_y - height/2
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non-maximum suppression to eliminate 
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #result = frame.copy()
    final_boxes = []
    crop_faces = []
    recog = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        final_boxes.append(box)

        # Extract position data
        left = int(box[0])
        top = int(box[1])
        width = int(box[2])
        height = int(box[3])

    # Draw bouding box with the above measurements
        cv2.rectangle(frame, (left,top), ((left+width), (top+height)), (0,255,255),2)
        # Display text about confidence rate above each box
        text = f'{confidences[i]:.2f}'
        cv2.putText(frame, text, (left+10, top-2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),1)

    # Crop faces
        face = frame[left:left+width, top:top+height]
        crop_faces.append(face)
    # Preprocess
        faces_edited = np.array(list(map(preprocess, crop_faces)))
    # Predict crop faces
        recog.append([loaded_model.predict([faces_edited])[0]])
        name = f'{recog[i]}'
        cv2.putText(frame, name, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),1)

    # Display text about number of detected faces on topleft corner
    total = f'number of detected faces: {len(final_boxes)}'
    cv2.putText(frame, total, (15, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),2,)
    cv2.imshow('face detection', frame)

    # frame is now the image capture by the webcam (one frame of the video)
    #cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
		# Break when pressing ESC
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()


