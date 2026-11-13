import sys
import cv2
import numpy as np

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)

img = cv2.imread("images/lane_test.jpg")
if img is None:
    print("이미지: 없음 (images/lane_test.jpg 를 넣어 주세요)")
else:
    print("이미지 크기(세로, 가로, 채널):", img.shape)
