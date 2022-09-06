import numpy as np

from pulse_estimation.approach import naive_ICA, naive_PCA

# face_cascade_file = "./resources/haar/haarcascade_frontalface_default.xml"
face_cascade_file = "./resources/haar/haarcascade_frontalface_alt.xml"
# target_video_file = "./resources/video/my/face_webcam_uncontrolled.mp4"  # Naiwna metoda wyboru komponentu NIE działa
# target_video_file = "./resources/video/my/face_controlled_not_diffused.mov" # Naiwna metoda wyboru komponentu NIE działa
target_video_file = "./resources/video/my/face_controlled_diffused.mov" # Naiwna metoda wyboru komponentu działa
# target_video_file = "./resources/video/other/face.mp4"  # Naiwna metoda wyboru komponentu działa

min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)

hr_low = 0.5
hr_high = 3

def main():
    naive_ICA(target_video_file, face_cascade_file, hr_low, hr_high, min_YCrCb, max_YCrCb, acc_hr=83.0/60)
    # naive_PCA(target_video_file, face_cascade_file, hr_low, hr_high, min_YCrCb, max_YCrCb)
    

    
