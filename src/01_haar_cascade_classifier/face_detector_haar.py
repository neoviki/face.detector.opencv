'''
    Face Detector based on HAAR Classifier

    Author  : Viki (a) Vignesh Natarajan
    Contact : vikiworks.io
'''
import cv2

def get_video_from_webcam():
    return cv2.VideoCapture(0)

#fname = ex.mp4
def get_video_from_file(fname):
    return cv2.VideoCapture(fname)


def detect_faces(gray_scale_image, algorithm):
    return algorithm.detectMultiScale(gray_scale_image, 1.1, 4)

def get_image_frame(video_stream):
    _, frame = video_stream.read()
    return frame

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def draw_bounding_box(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image

def display_image(image):
    cv2.imshow('img', image)

def is_exit_key_pressed():
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        return True
    else:
        return False

def main():
    #face classifier
    algorithm     = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video_stream  = get_video_from_webcam()

    while True:
        image_frame       = get_image_frame(video_stream)
        gray_scale_image  = get_grayscale(image_frame)
        faces             = detect_faces(gray_scale_image, algorithm)

        #image with bounding box
        final_image       = draw_bounding_box(image_frame, faces)
        display_image(final_image)

        if is_exit_key_pressed() == True:
            video_stream.release()
            break
 
main()
