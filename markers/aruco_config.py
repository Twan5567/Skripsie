
import cv2


ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))

CHARUCO_DICT_NAME = "DICT_5X5_100"
CHARUCO_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CHARUCO_DICT_NAME))

GCP_MARKER_IDS = list(range(0, 12))      
VEHICLE_MARKER_IDS = [20, 21]            
SCALE_BAR_MARKER_IDS = [30, 31]         

PLATE_MARKER_IDS = list(range(0, 8))

TEST_MARKER_IDS = [0]
FULL_MARKER_IDS = PLATE_MARKER_IDS


GCP_SIDE_MM_A4 = 150.0        
GCP_SIDE_MM_A3 = 220.0       
VEHICLE_SIDE_MM = 100.0      
PLATE_SIDE_MM = 250.0        
TEST_MARKER_SIDE_MM = 100.0  


MEASURED_SIDE_MM: dict[int, float] = {
    # 0: 149.7, 
}


def side_mm(marker_id: int, nominal: float) -> float:
    """Caliper-measured side length for `marker_id`, else the nominal value."""
    return MEASURED_SIDE_MM.get(marker_id, nominal)
