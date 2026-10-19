import cv2


ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))

TEST_MARKER_IDS = [0]


FULL_MARKER_IDS = list(range(8))


TEST_MARKER_SIDE_MM = 100.0  


FULL_MARKER_SIDE_MM = 250.0  


A4_SHEET_SIDE_MM = 184.57  
