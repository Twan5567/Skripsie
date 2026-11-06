import cv2

# the dictionary every detector in both pipelines uses: IDs 0-11 floor sheets,
# 20 the car's tag, 30-31 the scale bar
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
