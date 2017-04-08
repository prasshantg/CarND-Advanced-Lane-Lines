import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

left_line = Line()
right_line = Line()

def region_of_interest(img, vertices):
	"""
	Applies an image mask.

	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	#defining a blank mask to start with
	mask = np.zeros_like(img)

	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

	#filling pixels inside the polygon defined by "vertices" with the fill color
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

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

def apply_color_mask(img):
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Convert BGR to HSV
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# Extract yellow color lanes
	threshold_min = np.array([20, 80, 80], np.uint8)
	threshold_max = np.array([105, 255, 255], np.uint8)
	img_threshold = cv2.inRange(img_hsv, threshold_min, threshold_max);
	# Convert image to binary to highlight white lanes and avoid gray patches on road
	th, out_img = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV);
	out_img = cv2.bitwise_not(out_img)
	# Merge yellow extracted and binary image
	out_img = cv2.bitwise_or(out_img, img_threshold)
	return out_img

def apply_mag_threshold(img, kernel=3, m_threshold=(0,255), axis='x'):
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = img

	if axis == 'x':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel) #Take derivative in x
	if axis == 'y':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel) #Take derivative in y

	abs_sobel = np.absolute(sobel)
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

	sbinary = np.zeros_like(scaled_sobel)
	sbinary[(scaled_sobel >= m_threshold[0]) & (scaled_sobel <= m_threshold[1])] = 1

	return sbinary

def apply_color_threshold(img, c_threshold=(0,255), c_channel=2):
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

	c_channel = hls[:,:,c_channel]

	cbinary = np.zeros_like(c_channel)
	cbinary[(c_channel >= c_threshold[0]) & (c_channel <= c_threshold[1])] = 1

	return cbinary

def combined_thresholds(img, masked_img):
	s_binary = apply_mag_threshold(masked_img, 3, (10,150), 'y')
	c_binary = apply_color_threshold(img, (100,200), 2)

	color_binary = np.dstack((s_binary, np.zeros_like(s_binary), c_binary)) * 255

	combined_binary = np.zeros_like(s_binary)
	combined_binary[(s_binary == 1) | (s_binary == 1)] = 1

	return color_binary, combined_binary

def warp_image(img, src_corners, dst_corners):

	M = cv2.getPerspectiveTransform(src_corners, dst_corners)
	binary_warped = cv2.warpPerspective(img, M, (1280,720))

	return binary_warped

def detect_next_lane(img):
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100

	left_fit = left_line.current_fit
	right_fit = right_line.current_fit

	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
	# Again, extract left and right line pixel positions
	left_line.allx = nonzerox[left_lane_inds]
	left_line.ally = nonzeroy[left_lane_inds] 
	right_line.allx = nonzerox[right_lane_inds]
	right_line.ally = nonzeroy[right_lane_inds]
	# Fit a second order polynomial to each
	left_fit = np.polyfit(left_line.ally, left_line.allx, 2)
	right_fit = np.polyfit(right_line.ally, right_line.allx, 2)

	left_line.current_fit = left_fit
	right_line.current_fit = right_fit

	# Create an image to draw on and an image to show the selection window
	out_img = np.dstack((img, img, img))*255
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	return out_img

def detect_lanes(img):
	h_margin = 80
	margin = 80
	minpix = 50
	nwindows = 18

	histogram = np.sum(img[np.int(img.shape[0]/2):,:], axis=0)
	out_img1 = np.dstack((img, img, img))*255
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
	for win in range(nwindows - 2):
		lcurrent = leftx_current - h_margin
		rcurrent = rightx_current - h_margin
		window = win + 2
		good_left_inds = ((nonzerox > 0) & (nonzeroy > 0)).nonzero()[0]
		good_right_inds = ((nonzerox > 0) & (nonzeroy > 0)).nonzero()[0]
		for i in range(2):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = img.shape[0] - (window+1)*window_height
			win_y_high = img.shape[0] - window*window_height
			win_xleft_low = (lcurrent + i * h_margin) - margin
			win_xleft_high = (lcurrent + i * h_margin) + margin
			win_xright_low = (rcurrent + i * h_margin) - margin
			win_xright_high = (rcurrent + i * h_margin) + margin

			# Draw the windows on the visualization image
			cv2.rectangle(out_img1,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
			cv2.rectangle(out_img1,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

			# Identify the nonzero pixels in x and y within the window
			linds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
			rinds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
			if i == 0:
				good_left_inds = linds
				good_right_inds = rinds

			if len(linds) > len(good_left_inds):
				good_left_inds = linds
			if len(rinds) > len(good_right_inds):
				good_right_inds = rinds

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
	left_line.allx = nonzerox[left_lane_inds]
	left_line.ally = nonzeroy[left_lane_inds]
	right_line.allx = nonzerox[right_lane_inds]
	right_line.ally = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
	left_fit = np.polyfit(left_line.ally, left_line.allx, 2)
	right_fit = np.polyfit(right_line.ally, right_line.allx, 2)

	left_line.diffs = left_line.current_fit[len(left_line.current_fit)-1] - left_fit
	right_line.diffs = right_line.current_fit[len(right_line.current_fit)-1] - right_fit

	left_line.current_fit = left_fit
	right_line.current_fit = right_fit

	# Create an image to draw on and an image to show the selection window
	out_img = np.dstack((img, img, img))*255
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	left_line.detected = True
	right_line.detected = True

	return out_img, out_img1

def draw_lane_mask(orig_img, img, src_corners, dst_corners):
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(img).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	left_fit = left_line.current_fit
	right_fit = right_line.current_fit

	# Recast the x and y points into usable format for cv2.fillPoly()
	ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	Minv = cv2.getPerspectiveTransform(dst_corners, src_corners)
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (orig_img.shape[1], orig_img.shape[0])) 
	# Combine the result with the original image
	result_warped = cv2.addWeighted(orig_img, 1, newwarp, 0.3, 0)
	return result_warped

def calculate_radius(img):
	ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
	y_eval = np.max(ploty)

	left_fit = left_line.current_fit
	right_fit = right_line.current_fit

	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
	#print(left_curverad, right_curverad)
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(left_line.ally*ym_per_pix, left_line.allx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(right_line.ally*ym_per_pix, right_line.allx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	return left_curverad, right_curverad

def process_image(img, save_img=False, img_index=0):
	#Apply a distortion correction to raw images.
	undistort_img = undistort_image(img, mtx_l, dist_l)
	#Use color transforms, gradients, etc., to create a thresholded binary image.
	masked_img = apply_color_mask(undistort_img)
	if save_img == True:
		cv2.imwrite('output_images/masked_output{0}.jpg'.format(idx+1), masked_img)

	color_binary, binary_combined = combined_thresholds(undistort_img, masked_img)
	if save_img == True:
		cv2.imwrite('output_images/color_binary_output{0}.jpg'.format(idx+1), color_binary)
		cv2.imwrite('output_images/binary_combined_output{0}.jpg'.format(idx+1), binary_combined)

	#Apply a perspective transform to rectify binary image ("birds-eye view").
	src_corners = np.float32([[800, 470],[1210, 720],[550,470],[180,720]])
	dst_corners = np.float32([[1260,0],[1260,720],[40,0],[40,720]])

	orig_warped = warp_image(undistort_img, src_corners, dst_corners)
	out_img1 = cv2.resize(orig_warped, (0,0), fx=0.5, fy=0.5)
	cv2.putText(out_img1, "Perspective transform", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
	if save_img == True:
		cv2.imwrite('output_images/transformed_output{0}.jpg'.format(idx+1), orig_warped)

	#dst_corners = np.float32([[700,0],[700,1280],[20,0],[20,1280]])
	binary_warped_ro = warp_image(binary_combined, src_corners, dst_corners)
	color_warped_ro = warp_image(color_binary, src_corners, dst_corners)

	#Remove dashboard
	vertices = np.array([[(0,640),(0,0),(1280,0),(1280,640)]], dtype=np.int32)
	binary_warped = region_of_interest(binary_warped_ro, vertices)
	color_warped = region_of_interest(color_warped_ro, vertices)

	#Used only for drawing
	out_img2 = cv2.resize(color_warped, (0,0), fx=0.5, fy=0.5)
	cv2.putText(out_img2, "Threshold combined", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
	if save_img == True:
		cv2.imwrite('output_images/binary_warped_output{0}.jpg'.format(idx+1), binary_warped)
		cv2.imwrite('output_images/color_warped_output{0}.jpg'.format(idx+1), color_warped)

	if left_line.detected == False | right_line.detected == False:
		lane_detected, win_img = detect_lanes(binary_warped)
		if save_img == True:
			cv2.imwrite('output_images/lanes_detected_output{0}.jpg'.format(idx+1), lane_detected)
			cv2.imwrite('output_images/win_img_output{0}.jpg'.format(idx+1), win_img)

	if left_line.detected == True & right_line.detected == True:
		lane_detected = detect_next_lane(binary_warped)

	out_img3 = cv2.resize(lane_detected, (0,0), fx=0.5, fy=0.5)
	cv2.putText(out_img3, "Lanes detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

	leftrad, rightrad = calculate_radius(binary_warped)

	result = draw_lane_mask(undistort_img, binary_warped, src_corners, dst_corners)
	cv2.putText(result, "Left radius {0}m, Right radius {1}m".format(np.int(leftrad), np.int(rightrad)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
	out_img4 = cv2.resize(result, (0,0), fx=0.5, fy=0.5)

	if save_img == True:
		cv2.imwrite('output_images/final_output{0}.jpg'.format(idx+1), result)

	iresult1 = np.hstack((out_img1, out_img2))
	iresult2 = np.hstack((out_img3, out_img4))
	final = np.vstack((iresult1, iresult2))

	return final

#Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
mtx_l, dist_l = calibrate_camera()

test_images = glob.glob('test_images/test*.jpg')
test_images.sort()
for idx, fname in enumerate(test_images):
	img = cv2.imread(fname)
	result = process_image(img, True, idx)
	left_line.detected = False
	left_line.allx = None
	left_line.ally = None
	left_line.current_fit = [np.array([False])]
	right_line.detected = False
	right_line.allx = None
	right_line.ally = None
	right_line.current_fit = [np.array([False])]


challenge_output = 'project_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
challenge_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
challenge_clip.write_videofile(challenge_output, audio=False)
