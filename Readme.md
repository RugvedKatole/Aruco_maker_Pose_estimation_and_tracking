# Install Dependencies
run following command to install all dependencies

`pip3 install -r requirements.txt`

# 1: Calibration

## Step 1: Calibration board and images
Print Calibration.pdf exactly to 1:1 scale and glue it a flat surface

Click atleast 20 or more picture using the camera in a fix postion.
For instance check images in Calibration Images folder

Move all the images to Calibration_images folder

## Step 2: Obtain the Matrix Coeff and Distortion matrix

Run the Calibration.py file using following flags

`python3 Calibration.py --image_dir <path_to_calibration_images> --image_format <png/jpg> --prefix <fixed_image_prefix, for eg calibration_> --square_size <size of square in calirbation image, 0.022m in our case> --width 12 --height 8 --save_file ./calib.yml `

# 2: Start Tracking

## Step 1: Feed the parameters
Copy paste the matrix Coeff (K) and distortion matrix(d) from calib.yml into aruco_tracker.py 

## Step 2: Start Tracking

run the aruco_tracker.py to track the aruco markers

`python3 aruco_tracker.py`