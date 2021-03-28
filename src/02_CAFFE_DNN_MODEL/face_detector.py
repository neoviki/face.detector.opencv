'''
    Face Detector based on CAFFE Model ( DNN )

    Author  : Viki (a) Vignesh Natarajan
    Contact : vikiworks.io
'''

import cv2
import time
from imutils.video import VideoStream
import numpy as np
import imutils

neural_network = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

def get_video_from_webcam():
    return VideoStream(src=0).start()

def get_image_frame(video_stream):
    return video_stream.read()

def resize_image(image, width):
    return imutils.resize(image, width=width)


def image2blob(image):
    return cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

def detect_faces(image_blob):
    neural_network.setInput(image_blob)
    return neural_network.forward()

def add_label(image, label, x, y):
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

def draw_bounding_box(image_frame, faces):
    (h, w) = image_frame.shape[:2]
    for i in range(0, faces.shape[2]):
        accuracy = faces[0, 0, i, 2]

        if accuracy < 0.5:
            continue

        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        y = startY - 10
        if y < 10:
            y = startY + 10
        
        cv2.rectangle(image_frame, (startX, startY), (endX, endY),(0, 0, 255), 2)

        label = "{:.2f}%".format(accuracy * 100)
        add_label(image_frame, label, startX, y)

    return image_frame

def is_exit_key_pressed():
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        return True
    else:
        return False

def display_image(image):
    cv2.imshow('Frame', image)

def main():
    video_stream = get_video_from_webcam()
    time.sleep(1.0)

    while True:
        image_frame = get_image_frame(video_stream)
        image_frame = resize_image(image_frame, 400)

        blob = image2blob(image_frame)

        faces = detect_faces(blob)

        image_frame = draw_bounding_box(image_frame, faces)

        display_image(image_frame)

        if is_exit_key_pressed() == True:
            cv2.destroyAllWindows()
            video_stream.stop()

main()
