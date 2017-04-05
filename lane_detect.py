import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

sobel_kernel = 3
mag_threshold = (40, 130)
color_threshold = (100, 200)
color_channel = 2

def undistort_image(img, mtx, dist):
	dst_img = cv2.undistort(img, mtx, dist, None, mtx)
	return dst_img

def calibrate_camera():
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

	objpoints = [] #3d points in real world space
	imgpoints = [] #2d points in image plane

	#Make a list of calibration images
	images = glob.glob('camera_cal/calibration*.jpg')

	#Step through the list and search for chessboard corners
	for i, fname in enumerate(images):
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		#Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)

			if i == 0:
				cv2.imwrite('output_images/no_corners.jpg', img)
				cv2.drawChessboardCorners(img, (9,6), corners, ret)
				cv2.imwrite('output_images/corners_found.jpg', img)

	test_img = cv2.imread('camera_cal/calibration1.jpg')
	imgsize = (test_img.shape[1], test_img.shape[0])

	#Do camera calibration given object points and image points
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgsize, None, None)

	dst = cv2.undistort(test_img, mtx, dist, None, mtx)
	cv2.imwrite('output_images/undistort_output.jpg', dst)

	return mtx, dist

def apply_mag_threshold(img, kernel=3, m_threshold=(0,255), axis='x'):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	if axis == 'x':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel) #Take derivative in x
	if axis == 'y':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel) #Take derivative in y

	abs_sobel = np.absolute(sobel)
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

	sbinary = np.zeros_like(scaled_sobel)
	sbinary[(scaled_sobel >= m_threshold[0]) & (scaled_sobel <= mag_threshold[1])] = 1

	return sbinary

def apply_color_threshold(img, c_threshold=(0,255), c_channel=2):
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

	c_channel = hls[:,:,c_channel]

	cbinary = np.zeros_like(c_channel)
	cbinary[(c_channel >= c_threshold[0]) & (c_channel <= c_threshold[1])] = 1

	return cbinary

def combined_thresholds(img):
	s_binary = apply_mag_threshold(img, sobel_kernel, mag_threshold, 'y')
	c_binary = apply_color_threshold(img, color_threshold, color_channel)

	#color_binary = np.dstack((np.zeros_like(s_binary), s_binary, c_binary)) * 255
	color_binary = np.dstack((np.zeros_like(s_binary), np.zeros_like(s_binary), s_binary)) * 255

	combined_binary = np.zeros_like(s_binary)
	combined_binary[(s_binary == 1) | (c_binary == 1)] = 1

	return color_binary, combined_binary

def draw_lines(img):
	cv2.line(img, (770,480),(1180,720), [255,0,0],4)
	cv2.line(img, (580,480),(230,720), [255,0,0],4)

	return img

def warp_image(img, src_corners, dst_corners):

	M = cv2.getPerspectiveTransform(src_corners, dst_corners)
	binary_warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]))

	return binary_warped

def detect_lanes(img, margin=100, minpix=50, nwindows=18):
	histogram = np.sum(img[np.int(img.shape[0]/2):,:], axis=0)
	out_img = np.dstack((img, img, img))*255
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	window_height = np.int(img.shape[0]/nwindows)
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = img.shape[0] - (window+1)*window_height
		win_y_high = img.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)
	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	# Generate x and y values for plotting
	ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw on and an image to show the selection window
	#out_img = np.dstack((img, img, img))*255
	# Color in left and right line pixels
	#out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	#out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	return out_img, left_fitx, right_fitx, left_lane_inds, right_lane_inds, ploty

def draw_lanes(img, left_fitx, right_fitx, ploty, margin):
	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	window_img = np.zeros_like(img)
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))
	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(img, 1, window_img, 0.3, 0)

	return result

def draw_lane_mask(orig_img, img, left_fitx, right_fitx, ploty, src_corners, dst_corners):
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(img).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	Minv = cv2.getPerspectiveTransform(dst_corners, src_corners)
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
	# Combine the result with the original image
	result_warped = cv2.addWeighted(orig_img, 1, newwarp, 0.3, 0)
	return result_warped

def process_image(img):
	#Apply a distortion correction to raw images.
	undistort_img = undistort_image(img, mtx_l, dist_l)
	#Use color transforms, gradients, etc., to create a thresholded binary image.
	color_binary, binary_combined = combined_thresholds(undistort_img)

	#Used only for drawing
	#dst_img = draw_lines(undistort_img)

	#Apply a perspective transform to rectify binary image ("birds-eye view").
	src_corners = np.float32([[770, 480],[1180, 720],[580,480],[230,720]])
	dst_corners = np.float32([[1180,0],[1180,720],[230,0],[230,720]])
	binary_warped = warp_image(binary_combined, src_corners, dst_corners)

	#Used only for drawing
	out_img = np.dstack((np.zeros_like(binary_warped), binary_warped, np.zeros_like(binary_warped)))*255

	#Detect lane pixels and fit to find the lane boundary.
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	out_img1, left_fitx, right_fitx, left_lane_inds, right_lane_inds, ploty = detect_lanes(binary_warped, margin, minpix)

	#out_img = draw_lanes(binary_warped, left_fitx, right_fitx, ploty, margin)

	#out_img = draw_lane_mask(undistort_img, binary_warped, left_fitx, right_fitx, ploty, src_corners, dst_corners)

	return out_img

#Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
mtx_l, dist_l = calibrate_camera()

test_images = glob.glob('test_images/test*.jpg')
test_images.sort()

for idx, fname in enumerate(test_images):
	img = cv2.imread(fname)
	result = process_image(img)

#Determine the curvature of the lane and vehicle position with respect to center.
#Warp the detected lane boundaries back onto the original image.
#Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
#	cv2.imwrite('output_images/final_output{0}.jpg'.format(idx+1), out_img)

challenge_output = 'challenge_output.mp4'
clip1 = VideoFileClip("challenge_video.mp4")
challenge_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
challenge_clip.write_videofile(challenge_output, audio=False)
