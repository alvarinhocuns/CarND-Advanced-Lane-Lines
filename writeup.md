## Writeup Template

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

[image1]: ./output_images/chess.jpg "Comparison between the original and undistorted images"
[image2]: ./output_images/testUndistort.jpg "Road Transformed"
[image3]: ./output_images/testFilters.jpg "Filters and color example"
[image4]: ./output_images/warp.jpg "Warp example"
[image5]: ./output_images/slidingWindows.png "Sliding windows example"
[image6]: ./output_images/RadiusAndPosition.png "Radius and position of the car"
[video1]: ./projectlines_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is written in the function Calibrate() inside the utilities.py file, from lines 7 through 25.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The next figures are an example of how I applied the distortion correction to one of the test images.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The code is written in the functions abs_sobel_thresh(), mag_thresh(), dir_threshold(), S_threshold() and applyFilters(), from line 47 through 129 in the utilities.py file.  Here's an example of my output for this step, where the two contributions from the color (blue) and gradient filters (green) can be identified separately. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 28 through 35 in the file `utilities.py`.  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points by looking at four points of parallel lines in a test image.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 740,480       | 900, 0        | 
| 1012,650      | 900, 650      |
| 308,650       | 200, 650      |
| 550,480       | 200, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial

The code to identify the lines is written in two classes:
 * Line: That search for each individual line
 * Lines: That includes the two lines and performs the sanity checks and measurements involving the two lines together.

This code is implemented in the file Line.py.
The first time the class is looking for a line or after some frames didn't pass the sanity check, the script search for the lines using the sliding windows method. This method is called slidingWindows in the script and written between lines 39 and 79. In the next figure can be seen in the next image.

![alt text][image5]

If the lines where detected correctly in the last frame I use the last position to search for the lines in the current frame.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius using the method getRadius() from the Line class. This method calculates the radius of curvature in the current frame to check the current measurement, and also measures the radius using the last n frames in order to smooth the measurement.
The position of the vehicle is calculated measuring the distance to the two lines from the center of the image and assuming that the distance between lines is 3.7m. This is done in the method getCarPosition() from the Lines class in the Line.py file.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the class processImage in the file LineFinding.py.Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In some frames with shadows and different road conditions the pipeline is likely to fail. To make it perform better I should try to tune a little bit more the filters in order to discriminate other objects.
I performed a sanity check that checks the next conditions:
 * The two lines must be more or less parallel
 * The car must be located between the two lines. This is an ad hoc condition.
 * The line separation must be +-3.7m

If the measurement doesn't meet this conditions I consider that the lines were not detected and they are not saved. If 16 of the last 25 measurements were not saved, the pipeline is reset and in the next iteration it must try to find the lines using the sliding windows.

