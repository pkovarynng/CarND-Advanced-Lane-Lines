import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def calibrateCamera():
    nx = 9 # the number of inside corners in x
    ny = 6 # the number of inside corners in y

    # Read in and make a list of calibration images
    images = glob.glob('camera_cal\calibration*.jpg')

    # Arrays to store object points and image points from all the images
    objpoints = [] # 3D points in real world space, z = 0 for every point
    imgpoints = [] # 2D points in image plane

    # Prepare object points, like (0,0,0), (1,0,0), ... (nx-1,ny-1,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates

    shape = (0, 0)
    for fname in images:
        # Read in current calibration image
        img = mpimg.imread(fname)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Store the shape for using it outside the loop
        shape = gray.shape[::-1]
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, append to the object points and to the image points
        if ret == True:
            objpoints.append(objp) # Appending the same for each image
            imgpoints.append(corners)
            
            ## Draw and display the corners
            #img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #plt.imshow(img)
            #plt.show()
        else:
            print("Finding chessboard corners failed for:", fname)

    # Get the camera calibration matrix and distortion coefficients based on object and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    
    return ret, mtx, dist

def testCalibration(mtx, dist):
    # Test the camera calibration on one of the images provided for calibration
    # (There was no chessboard image provided explicitly for testing the calibration,
    # but findChessboardCorners failed for this image, so it was actually NOT used
    # in the calibration process.
    test_img = mpimg.imread('camera_cal\calibration1.jpg')
    undst_test_img = cv2.undistort(test_img, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(test_img, cmap='gray')
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(undst_test_img, cmap='gray')
    ax2.set_title('Undistorted Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    
# Calibrate camera: get the camera calibration matrix and distortion coefficients
ret, mtx, dist = calibrateCamera()

# Test the calibration
testCalibration(mtx, dist)

