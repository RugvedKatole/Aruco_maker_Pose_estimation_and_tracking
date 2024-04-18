#!/usr/bin/env python3

import numpy as np
import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)  # Get the camera source

def track(matrix_coefficients, distortion_coefficients,square_size):
    while True:
        ret, frame = cap.read()
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)  # Use 5x5 dictionary to find markers
        parameters = aruco.DetectorParameters()  # Marker detection parameters
        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                parameters=parameters)
        if np.all(ids is not None):  # If there are markers found by detector
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], square_size, matrix_coefficients,
                                                                           distortion_coefficients)
                print(rvec,tvec)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis
        # Display the resulting frame
        # For full screen uncomment line 27, 28
        # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('frame', frame)
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    matrix_coefficients = np.array([[ 4.6223891462597084e+02, 0., 6.6430416829945170e+02],[ 0.,
       4.5939142549071386e+02, 3.1840511295018064e+02], [0., 0., 1. ]])
    distortion_coefficients = np.array([ 2.4925430862213918e-01, -4.0709477553344736e-01,
       -6.7079559991616222e-03, 2.3539216376681463e-03,
       2.0450400190251042e-01 ])
    track(matrix_coefficients,distortion_coefficients)
    