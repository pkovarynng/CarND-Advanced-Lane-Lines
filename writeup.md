## Udacity CarND Project 2 - Advanced Lane Finding

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration-test.jpg "Undistorted"
[image2]: ./output_images/undistorted_test2.jpg "Road Transformed"
[image3]: ./output_images/binary_combo_test2.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines1.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/result_test2.jpg "Output"
[video1]: ./project_video_result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

This is my Writeup, please carry on reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrateCamera()` function in the file called `p2.py` from line 41 through 85.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

The `cv2.findChessboardCorners()` function failed for three of the calibration images. Could be that they don't have the sufficient white border around the edges.

The camera calibration step returns the following important pieces of information that will be used in the first step of my image processing pipeline, the distortion correction:
1. calibration matrix
2. distortion coefficients

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I include the following distortion-corrected test image:

![alt text][image2]

The distortion-correction is performed using the `cv2.undistort()` function. The call to this OpenCV function can be found in my `process_image()` function in line 366 in the `p2.py` file. It uses the camera calibration matrix and the distortion coefficients returned from the camera calibration.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The thresholding steps can be found in the `p2.py` file at lines 105 through 202. There are individual functions for each thresholding method and there is the function `get_binary_image()` that calls the others and combines their result into a single binary image.

Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform can be found in the `process_image()` function from lines 341 through 383 in the file `p2.py`. The perspective transform is performed on the binary combo image based on the source (`srcpoints`) and destination (`dstpoints`) points.  I chose to hardcode the source and destination points in the following manner:

```python
    # 2) Define 4 source points
    srcpoints = [[190, imshape[0]],
                [imshape[1]//2-45, 450],
                [imshape[1]//2+45, 450],
                [imshape[1]-160, imshape[0]]]
    # 3) Define 4 destination points
    dstpoints = [[330, imshape[0]],
                [330, 0],
                [950, 0],
                [950, imshape[0]]]
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 330, 720      | 
| 595, 450      | 330, 0        |
| 685, 450      | 950, 0        |
| 1120, 720     | 950, 720      |

I verified that my perspective transform was working as expected by drawing `srcpoints` and `dstpoints` onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In line 400 in my `p2.py` I call my `fit_polynomial()` function that fits a 2nd order polynomial on the left and right lane lines. This is done by collecting the pixel points of the lines first, then calling the `np.polyfit()` function for each of the two arrays of pixel points separately. To illustrate the result of this, I inserted the following image:

![alt text][image5]

The yellow lines are the fitted polynomials. The pixel points of the left and the right lanes that were considered for the polynomials are in red and in blue, respectively. The points in the above image were collected using the sliding window technique learned in this curse.

The sliding window technique uses a histogram. In addition to using only the bottom half of the binary image for the histogram, as I saw in the lectures, I used a side margin also. This can be seen from line 225 through 227 in the file `p2.py` in the `find_lane_pixels()` function.

When processing the video, the sliding window technique is used only for the first frame. From the next frame on, the pixel points for the new polynomials are collected from a hard-coded margin of 50 pixels around the previous polynomials. This is implemented by the `search_around_poly()` function in my `p2.py` file.

The decision about which of the two functions mentioned above should be called is made in function `get_lane_line_pixels` in my `p2.py` file.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

##### Radius of the curvature

I did this in lines 433 through 439 in my code in `p2.py`.

This is done in real space, not in pixel space. So the `measure_curvature_real()` uses the pixel to meter conversions as advised. This function calculates the current radiuses based on the fitted polynomials and the learned formula. The radiuses are calculated for the coordinates of the points where the polynomials reach the bottom of the image (they are the closest to the car).

On top of calculating the current radiuses for the current polynomials, my code stores the radiuses of maximum 75 previous iterations and calculates their average. This is done for the left and the right lines separately. So there will be two average radiuses which then will be averaged again. The result of those avarages can be seen in the video.

##### Position of the vehicle

This is done in lines 441 through 458 in my code in `p2.py`, so just after radius of the curvature.

This one is also done in real space. So the same pixel to meter conversions are applied as for radius of the curvature.

For the calculation it is assumed that the middle of the image matches on the middle axis of the car. Therefore if the car is exactly in the middle of the lane, then the image midpoint should be exactly halfway between the base points (with the highest y coordinate) of the two lane lines. If not, then the difference between the two gives the offset of the car.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 400 through 430 in my code in `p2.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I can see from trying out my pipeline on the challenage videos that implementing some kind of sanity check on the polynomials could improve the results. If the sanity check fails, the polynomial should be searched from scratch, using the sliding window technique.

Averaging the polynomial could smooth the little jittering of the edges of the lane drawn on the original images.

Now my pipeline is more robust than the one in the introductory project, but it can still be fooled by different colors on and around the lane. Further experimenting with ways to combine the various thresholding techniques and filtering colors may also be beneficial.
