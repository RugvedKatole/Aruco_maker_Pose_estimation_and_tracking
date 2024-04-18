#!/usr/bin/env python3

import os
import numpy as np
import cv2
import cv2.aruco as aruco
import rospy 
from geometry_msgs.msg import Pose, Point, Quaternion
import argparse
import yaml

def read_calibration_file(calibration_file):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

def track(matrix_coefficients, distortion_coefficients, square_size):
    rospy.init_node('aruco_pose_publisher', anonymous=True)  # Initialize ROS node
    pose_publisher = rospy.Publisher('aruco_pose', Pose, queue_size=10)  # Create ROS publisher

    cap = cv2.VideoCapture(0)  # Get the camera source

    while not rospy.is_shutdown():  # ROS-compatible loop
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        parameters = aruco.DetectorParameters()

        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if np.all(ids is not None): 
            for i in range(0, len(ids)):
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], square_size, matrix_coefficients, distortion_coefficients)

                # Create ROS Pose message
                pose_msg = Pose()
                pose_msg.position = Point(*tvec[0][0])  # Set position (from tvec)

                # Convert rvec to Quaternion for orientation
                rvec_matrix = cv2.Rodrigues(rvec)[0] 
                quaternion = cv2.RQDecomp3x3(rvec_matrix)[1]
                pose_msg.orientation = Quaternion(*quaternion) 

                pose_publisher.publish(pose_msg)  # Publish the pose

                aruco.drawDetectedMarkers(frame, corners) 
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  

        cv2.imshow('frame', frame)
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArUco marker pose estimation with ROS integration.')
    parser.add_argument('--square_size', type=float, help='Size of the ArUco marker (in meters)')
    parser.add_argument('--calibration_file', type=str, help='Path to the YAML calibration file')
    args = parser.parse_args()
    directory = os.path.dirname(__file__)
    matrix_coefficients, distortion_coefficients = read_calibration_file(f"{directory}/{args.calibration_file}")

    track(matrix_coefficients, distortion_coefficients, args.square_size)
