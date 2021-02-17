import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # max number of iterations over which avarage is calculated
        self.n = 75 # 3 seconds (assumed frame rate is 25 frames/sec)
        # last n radians of curvature
        self.curverads = []
        # current fitting polynomial
        self.current_fit = [np.array([False])]
    
    def clear(self):
        self.detected = False  
        self.curverads = []
        self.current_fit = [np.array([False])]
        
    def append_curverad(self, curverad):
        self.curverads.append(curverad)
        if len(self.curverads) > self.n:
            self.curverads.pop(0)
            
    def get_avg_curverad(self):
        return int(sum(self.curverads) / len(self.curverads))
            
lline = Line()
rline = Line()

### FUNCTION DEFINITIONS BEGIN

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
        else:
            print("Finding chessboard corners failed for:", fname)

    # Get the camera calibration matrix and distortion coefficients based on object and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    
    return ret, mtx, dist

def showCalibrationResult(mtx, dist):
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

# # Show the calibration result
# showCalibrationResult(mtx, dist)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)    
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    #    is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_mag = np.uint8(255*mag/np.max(mag))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_mag)
    binary_output[(scaled_mag >= mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

# Thresholds the S-channel of HLS
# Exclusive lower (>) and inclusive upper (<=) bounds used
def s_threshold(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

# For testing
def show(title1, img1, title2, img2, cmap1=None, cmap2='gray'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1, cmap=cmap1)
    ax1.set_title(title1, fontsize=20)
    ax2.imshow(img2, cmap=cmap2)
    ax2.set_title(title2, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def get_binary_image(dst):
    # Sobel kernel size: odd number, in order to smooth gradient measurements
    ksize = 7

    # Apply thresholding functions on the undistorted image
    gradx = abs_sobel_thresh(dst, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(dst, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(dst, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(dst, sobel_kernel=ksize, thresh=(0.7, 1.3))
    s_binary = s_threshold(dst, thresh=(90, 255))
    
    # Combine the thresholding results
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

    # # for testing only:
    # show('Original Image', img, 'Undistorted Thresholded Binary', combined)
    # show('Original Image', img, 'Undistorted Thresholded Binary', s_binary)
    return combined
        
def hist(img):
    # Grab only the bottom half of the image and crop the sides also
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # Sum across image pixels vertically
    # The highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    # # Show the histogram - only for testing
    # plt.plot(histogram)
    # plt.show()
    
    # Return the histogram
    return histogram

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = hist(binary_warped)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    side_margin = 150
    leftx_base = np.argmax(histogram[side_margin:midpoint]) + side_margin
    rightx_base = np.argmax(histogram[midpoint:histogram.shape[0]-side_margin]) + midpoint
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # The width of the margin around the previous polynomial to search
    margin = 50

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### The area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = ((nonzerox > (lline.current_fit[0]*(nonzeroy**2) + lline.current_fit[1]*nonzeroy + 
                    lline.current_fit[2] - margin)) & (nonzerox < (lline.current_fit[0]*(nonzeroy**2) + 
                    lline.current_fit[1]*nonzeroy + lline.current_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (rline.current_fit[0]*(nonzeroy**2) + rline.current_fit[1]*nonzeroy + 
                    rline.current_fit[2] - margin)) & (nonzerox < (rline.current_fit[0]*(nonzeroy**2) + 
                    rline.current_fit[1]*nonzeroy + rline.current_fit[2] + margin)))
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Return new pixel positions for polynomials
    return leftx, lefty, rightx, righty

def get_lane_line_pixels(binary_warped):
    if lline.detected == True:
        # Search for pixels around the previous polynomial
        return search_around_poly(binary_warped)
    else:
        # Find our lane pixels first
        return find_lane_pixels(binary_warped)

def fit_polynomial(binary_warped, ym_per_pix, xm_per_pix):

    leftx, lefty, rightx, righty = get_lane_line_pixels(binary_warped)
    
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Generate y values
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    return ploty, left_fit_cr, right_fit_cr

def measure_curvature_real(binary_warped):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    ploty, left_fit, right_fit = fit_polynomial(binary_warped, ym_per_pix, xm_per_pix)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad

def process_image(img):
    '''
    Implements the pipeline - processes one image/frame
    '''
    # Undistort the image: gets the so called destination image
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # Get the combined binary image using gradient/color thresholding
    combined = get_binary_image(dst)
    
    # Prepare for warping the image
    # 1) Get image shape
    imshape = combined.shape
    # 2) Define 4 source points
    srcpoints = [[190, imshape[0]],
                [imshape[1]//2-45, 450],
                [imshape[1]//2+50, 450],
                [imshape[1]-160, imshape[0]]]
    # 3) Define 4 destination points
    dstpoints = np.float32([[330, imshape[0]],
                            [330, 0],
                            [950, 0],
                            [950, imshape[0]]])
    # 4) Get the transformation matrix for the perspective transform
    M = cv2.getPerspectiveTransform(np.float32(srcpoints), dstpoints)

    # Warp the combined binary image
    warped_combined = cv2.warpPerspective(combined, M, (dst.shape[1], dst.shape[0]), flags=cv2.INTER_LINEAR)
    #mpimg.imsave("my_warped_example.jpg", warped_combined, cmap="gray")

    ### DRAWING ####
    # In pixel space
    ploty, left_fit, right_fit = fit_polynomial(warped_combined, 1, 1)

    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_combined).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Get the inverse transformation matrix for the perspective transform
    Minv = cv2.getPerspectiveTransform(dstpoints, np.float32(srcpoints))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(dst, 1, newwarp, 0.3, 0)
    
    ### ADD TEXT TO IMAGE
    
    ### Curvature radius in real space
    left_curverad, right_curverad = measure_curvature_real(warped_combined)  
    lline.append_curverad(left_curverad)
    rline.append_curverad(right_curverad)
    avg_curverad = int((lline.get_avg_curverad() + rline.get_avg_curverad()) / 2)
    curvrad_text = 'Radius of curvature: {} m'.format(avg_curverad)
    cv2.putText(result, curvrad_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    ### Vehicle position
    histogram = hist(warped_combined)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    side_margin = 150
    leftx_base = np.argmax(histogram[side_margin:midpoint]) + side_margin
    rightx_base = np.argmax(histogram[midpoint:histogram.shape[0]-side_margin]) + midpoint
    midx_base = (leftx_base + rightx_base)//2
    xm_per_pix = 3.7/700
    offset = int((midx_base - midpoint) * xm_per_pix * 100)
    side = ''
    if offset > 0:
        side = 'left'
    else:
        side = 'right'
    offset_text  = 'Vehicle is {} cm {} of center'.format(abs(offset), side)
    cv2.putText(result,  offset_text, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Tracking
    lline.current_fit = left_fit
    lline.detected = True
    rline.current_fit = right_fit
    rline.detected = True
    
    # Return the result image
    return result

def test_on_images():
    '''
    Tests the pipeline on the provided test images
    '''
    images = glob.glob('test_images/*.jpg')
    for fname in images:
        # Read in the test image
        img = mpimg.imread(fname)
        # Process the image
        result = process_image(img)
        # Save the result in the output_images directory
        filename = os.path.basename(fname)
        mpimg.imsave("output_images/result_"+filename, result)
        # Show the result
        plt.imshow(result)
        plt.show()
        # Forget stored image data
        lline.clear()
        rline.clear()

def test_on_video(name):
    '''
    Runs the pipeline on a video and generates a result video
    '''
    input = name+'.mp4'
    white_output = name+'_result.mp4'
    clip1 = VideoFileClip(input)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)    

### FUNCTION DEFINITIONS END

#test_on_images()
test_on_video("project_video")
