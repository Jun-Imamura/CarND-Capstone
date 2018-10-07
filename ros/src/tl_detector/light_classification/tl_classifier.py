from styx_msgs.msg import TrafficLight
import numpy as np
import cv2

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # convert to HSV color space
        # https://jp.123rf.com/photo_65990122_%E8%89%B2%E3%83%9B%E3%82%A4%E3%83%BC%E3%83%AB%E5%90%8D%E5%BA%A6-rgb-hsb-hsv-%E8%89%B2%E7%9B%B8.html
        hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        SAT_MIN = 100
        SAT_MAX = 255
        VAL_MIN = 100
        VAL_MAX = 255
        THRESH_TO_JUDGE = 50

        # for red signal
        RED_HUE_MIN1 = np.array([  0.0/360*255, SAT_MIN, VAL_MIN],np.uint8)
        RED_HUE_MAX1 = np.array([ 20.0/360*255, SAT_MAX, VAL_MAX],np.uint8)        
        RED_HUE_MIN2 = np.array([330.0/360*255, SAT_MIN, VAL_MIN],np.uint8)
        RED_HUE_MAX2 = np.array([360.0/360*255, SAT_MAX, VAL_MAX],np.uint8)
        red_val1 = cv2.inRange(hsv_img, RED_HUE_MIN1, RED_HUE_MAX1) 
        red_val2 = cv2.inRange(hsv_img, RED_HUE_MIN2, RED_HUE_MAX2) 
        if cv2.countNonZero(red_val1) + cv2.countNonZero(red_val2) > THRESH_TO_JUDGE:
            return TrafficLight.RED

        # for yellow signal
        YLW_HUE_MIN1 = np.array([ 40.0/360*255, SAT_MIN, VAL_MIN],np.uint8)
        YLW_HUE_MAX1 = np.array([ 70.0/360*255, SAT_MAX, VAL_MAX],np.uint8)
        ylw_val1 = cv2.inRange(hsv_img, YLW_HUE_MIN1, YLW_HUE_MAX1)
        if cv2.countNonZero(ylw_val1) > THRESH_TO_JUDGE:
            return TrafficLight.YELLOW

        # for green signal
        GRN_HUE_MIN1 = np.array([ 90.0/360*255, SAT_MIN, VAL_MIN],np.uint8)
        GRN_HUE_MAX1 = np.array([140.0/360*255, SAT_MAX, VAL_MAX],np.uint8)
        grn_val1 = cv2.inRange(hsv_img, GRN_HUE_MIN1, GRN_HUE_MAX1)
        if cv2.countNonZero(grn_val1) > THRESH_TO_JUDGE:
            return TrafficLight.GREEN

        return TrafficLight.UNKNOWN
