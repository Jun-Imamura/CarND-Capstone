#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import numpy as np
from math import *

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size = 1)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''

        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #return light.state
        
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        ## Cropping Size
        crop_w = 1.0 #[m]
        crop_h = 2.0 #[m]
        image_h, image_w, _ = cv_image.shape

        tl_pos = light.pose.pose.position
        tl_pos = np.array([tl_pos.x, tl_pos.y, tl_pos.z])
        xTL, yTL = self.cvt_world_to_image(tl_pos,  crop_w/2,  crop_h/2)
        xBR, yBR = self.cvt_world_to_image(tl_pos, -crop_w/2, -crop_h/2)

        xTL = max(0, min(xTL, image_w-1))
        yTL = max(0, min(yTL, image_h-1))
        xBR = max(0, min(xBR, image_w-1))
        yBR = max(0, min(yBR, image_w-1))
        #rospy.logwarn("xTL, yTL, xBR, yBR = %s:%s:%s:%s", xTL, yTL, xBR, yBR)
        
        if xBR - xTL < 10 or yBR - yTL < 20:
            # cropped region too small
            return TrafficLight.UNKNOWN

        cv_image = cv_image[yTL:yBR, xTL:xBR]

        #cv2.imwrite("test.bmp", cv_image)   #for debug
        #Get classification
        return self.light_classifier.get_classification(cv_image)
        

    def cvt_world_to_image(self, wpos, offset_y, offset_z):
        cpos = self.cvt_world_to_car(wpos)
        cpos[1] += offset_y
        cpos[2] += offset_z
        return (self.cvt_car_to_image(cpos)).astype(int)


    def cvt_world_to_car(self, wpos):
        ## Convert world coordinate into car-relative coordinate
        wpos = np.append(wpos, 1)
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")        

        ## Ideally, pitch/roll dynamics should be considered.
        ## But it seems working only taking yaw into consideration
        yaw = tf.transformations.euler_from_quaternion(rot)[2] ## [roll, pitch, yaw] 

        mat = np.array([[cos(yaw), -sin(yaw), 0, trans[0]],
                        [sin(yaw),  cos(yaw), 0, trans[1]],
                        [       0,         0, 1, trans[2]],
                        [       0,         0, 0,        1]])

        #rospy.logwarn("wpos: cpos [%s:%s]", wpos, np.dot(mat, wpos)[0:3])
        return np.dot(mat, wpos)[0:3]

    def cvt_car_to_image(self, cpos):
        ## Convert car-relative coordinate to camera image coordinate 
        cpos = np.append(cpos, 1)
        
        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_w = self.config['camera_info']['image_width']
        image_h = self.config['camera_info']['image_height']

        cx = image_w/2
        cy = image_h/2
        cpos[2] -= 1.0
        ## For perspective transform
        mat1 = np.array([[fx, 0, cx, 0],
                         [ 0,fy, cy, 0],
                         [ 0, 0,  1, 0],
                         [ 0, 0,  0, 1]])
        ## For direction alignment (90deg in z-axis then 90deg in x-axis)
        mat2 = np.array([[ 0,-1,  0, 0],
                         [ 0, 0, -1, 0],
                         [ 1, 0,  0, 0],
                         [ 0, 0,  0, 1]])
        mat = np.dot(mat1, mat2)
        ret = np.dot(mat, cpos)
        ## Light center axis is not an image center.
        ## so I applied offset to the cropped region.
        return np.array([ret[0]/ret[2]-30, ret[1]/ret[2]+350])

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

        #TODO find the closest visible traffic light (if one exists)
        diff = len(self.waypoints.waypoints)
        for i, light in enumerate(self.lights):
            # Get stop line waypoint index
            line = stop_line_positions[i]
            temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
            # find closest stop line waypoint index
            d = temp_wp_idx - car_wp_idx
            if d >= 0 and d < diff:
                diff = d
                closest_light = light
                line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            #rospy.logwarn("light: %s", state)  # detected light color
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
