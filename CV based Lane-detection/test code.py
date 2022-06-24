
'''
# 이미지에 undistortion 테스트
img = cv2.imread('./calibration image/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])
plt.imshow(img)
print(img.shape[2])
print(img.shape[1])
print(img.shape[0])
resized_img = np.array(img.resize((960, 540)))  # width, height
plt.imshow(resized_img)
'''


'''
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])


# Visualize unwarp
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.05)
ax1.imshow(exampleImg_undistort)
x = [src[0][0],src[2][0],src[3][0],src[1][0],src[0][0]]
y = [src[0][1],src[2][1],src[3][1],src[1][1],src[0][1]]
ax1.plot(x, y, color='#33cc99', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
ax1.set_ylim([h,0])
ax1.set_xlim([0,w])
ax1.set_title('Undistorted Image', fontsize=30)
ax2.imshow(exampleImg_unwarp)
ax2.set_title('Unwarped Image', fontsize=30)
plt.show()
'''
# exampleImg_out2 = draw_data(exampleImg_out1, (rad_l + rad_r) / 2, d_center)
'''
# Set up plot
fig, axs = plt.subplots(1, 2, figsize=(20, 5))
fig.subplots_adjust(hspace=.2, wspace=.001)
axs = axs.ravel()

axs[0].imshow(exampleImg_out1)
axs[0].axis('off')
axs[1].imshow(exampleImg_out2)
axs[1].axis('off')

plt.show()
'''


'''
################################################# 잘되는지 확인 코드 Visualize undistortion###################################
# 파이프라인을 작동시켜 시연할 이미지 출력한다.  #-------------------calibration 잘 되는지 확인한다. --------------
exampleImg = cv2.imread('./calibration image/calibration1.jpg')
exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)
exampleImg_undistort = undistort(exampleImg)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
f.subplots_adjust(hspace=.2, wspace=.05)
plt.imshow(exampleImg)
ax1.imshow(exampleImg)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(exampleImg_undistort)
ax2.set_title('Undistorted Image', fontsize=30)

exampleImg_unwarp, M, Minv = unwarp(exampleImg_undistort, src, dst)
'''

#----------------------------------------테스트 시작------------------------------------#
'''
# Make a list of example images
images = glob.glob('./test image_line/*.jpg')

# Set up plot
fig, axs = plt.subplots(len(images), 2, figsize=(10, 20))
fig.subplots_adjust(hspace=.2, wspace=.001)
axs = axs.ravel()

i = 0

for image in images:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_bin, Minv = pipeline(img)
    axs[i].imshow(img)
    axs[i].axis('off')
    i += 1
    axs[i].imshow(img_bin, cmap='gray')
    axs[i].axis('off')
    i += 1
plt.show()
'''


#----------------------------------------테스트------------------------------------#
'''
# visualize the result on example image
exampleImg = cv2.imread('./test image_line/solidWhiteRight.jpg')
exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)
exampleImg = cv2.resize(exampleImg, dsize=(1280,720), interpolation=cv2.INTER_AREA)


# pipline을 통해 thresold를 적용하고 perspective transformation 한 이미지를 받는다.
exampleImg_bin, Minv = pipeline(exampleImg)
plt.imshow(exampleImg_bin)
plt.show()

# sliding window 함수에 적용한다.
left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = sliding_window_polyfit(exampleImg_bin)

h = exampleImg.shape[0]
left_fit_x_int = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
right_fit_x_int = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]
# print('fit x-intercepts:', left_fit_x_int, right_fit_x_int)

rectangles = visualization_data[0]
histogram = visualization_data[1]


# 선을 그린 결과 이미지를 생성하고 결과를 시각화한다.
out_img = np.uint8(np.dstack((exampleImg_bin, exampleImg_bin, exampleImg_bin)) * 255)
# plotting을 위한 x,y값을 generate한다.
ploty = np.linspace(0, exampleImg_bin.shape[0] - 1, exampleImg_bin.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

for rect in rectangles:
    # 시각화 이미지에 창을 그린다.
    cv2.rectangle(out_img, (rect[2], rect[0]), (rect[3], rect[1]), (0, 255, 0), 2)
    cv2.rectangle(out_img, (rect[4], rect[0]), (rect[5], rect[1]), (0, 255, 0), 2)
# Identify the x and y positions of all nonzero pixels in the image
# 이미지에서 0이 아닌 픽셀값들의 x, y 포지션을 확인한다.
nonzero = exampleImg_bin.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100, 200, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)

# sliding window polyfit을 적용한 이미지로부터 히스토그램을 출력한다
plt.plot(histogram)
plt.xlim(0, 1280)
plt.show()
#----------------------------------------테스트------------------------------------


# visualize the result on example image
exampleImg2 = cv2.imread('./test image_line/solidWhiteCurve.jpg')
exampleImg2 = cv2.cvtColor(exampleImg2, cv2.COLOR_BGR2RGB)
exampleImg2 = cv2.resize(exampleImg2, dsize=(1280,720), interpolation=cv2.INTER_AREA)

exampleImg2_bin, Minv = pipeline(exampleImg2)
margin = 80

left_fit2, right_fit2, left_lane_inds2, right_lane_inds2 = polyfit_using_prev_fit(exampleImg2_bin, left_fit, right_fit)

# Generate x and y values for plotting
ploty = np.linspace(0, exampleImg2_bin.shape[0] - 1, exampleImg2_bin.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
left_fitx2 = left_fit2[0] * ploty ** 2 + left_fit2[1] * ploty + left_fit2[2]
right_fitx2 = right_fit2[0] * ploty ** 2 + right_fit2[1] * ploty + right_fit2[2]

# 이미지에 선을 그리고 선택 창 영역을 보여준다.
out_img = np.uint8(np.dstack((exampleImg2_bin, exampleImg2_bin, exampleImg2_bin)) * 255)
window_img = np.zeros_like(out_img)

# 좌우 라인 픽셀에 색깔 입히기
nonzero = exampleImg2_bin.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
out_img[nonzeroy[left_lane_inds2], nonzerox[left_lane_inds2]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds2], nonzerox[right_lane_inds2]] = [0, 0, 255]

# polygon을 생성하여 검색 창 영역(OLD FIT)을 보여 주고 x 및 y 점을 cv2fillPoly()에 사용할 수 있는 형식으로 재구성합니다.
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# wraped 이미지에 선을 그린다.
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.imshow(result)
plt.plot(left_fitx2, ploty, color='yellow')
plt.plot(right_fitx2, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()
#----------------------------------------테스트 종료------------------------------------
'''

import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# 이미지 리스트 만들고 출력하기
images = glob.glob('./calibration image/calibration*.jpg')

fig, axs = plt.subplots(5, 4, figsize=(16, 11))
fig.subplots_adjust(hspace=.2, wspace=.001)
axs = axs.ravel()


# 체스보드에서 코너 탐색 시작
for i, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 코너를 찾는다
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # 찾을경우 object points 와 image points에 추가한다.
    if ret == True:
        objpoints.append(objp)

        # this step to refine image points was taken from:
        # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        axs[i].axis('off')
        axs[i].imshow(img)

'''
# 이미지에 undistortion 테스트
img = cv2.imread('./calibration image/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])
plt.imshow(img)
print(img.shape[2])
print(img.shape[1])
print(img.shape[0])
resized_img = np.array(img.resize((960, 540)))  # width, height
plt.imshow(resized_img)
'''
img = cv2.imread('./calibration image/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# 위에서 실행한 camera calibration 값들을 다른 변수에 저장하고 실행한다./cv2이용한다.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
dst = cv2.undistort(img, mtx, dist, None, mtx)

# 나중에 사용하기 위해 calibration값을 저장해둔다. (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("calibration.p", "wb"))
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)


# undistort image using camera calibration matrix from above
def undistort(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

'''
################################################# 잘되는지 확인 코드 Visualize undistortion###################################
# 파이프라인을 작동시켜 시연할 이미지 출력한다.  #-------------------calibration 잘 되는지 확인한다. --------------
exampleImg = cv2.imread('./calibration image/calibration1.jpg')
exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)
exampleImg_undistort = undistort(exampleImg)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
f.subplots_adjust(hspace=.2, wspace=.05)
plt.imshow(exampleImg)
ax1.imshow(exampleImg)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(exampleImg_undistort)
ax2.set_title('Undistorted Image', fontsize=30)

exampleImg_unwarp, M, Minv = unwarp(exampleImg_undistort, src, dst)
'''

##############################################Perspective transforamtion###################################################################

def unwarp(img, src, dst):

    h, w = img.shape[:2]  # 밑에 w 인식을 못해서 다시 씀
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() wrap이미지를 top-town 관점에서 실행한다.
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    plt.imshow(warped)
    return warped, M, Minv

 # h,w = exampleImg_undistort.shape[:2]


'''
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])


# Visualize unwarp
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.05)
ax1.imshow(exampleImg_undistort)
x = [src[0][0],src[2][0],src[3][0],src[1][0],src[0][0]]
y = [src[0][1],src[2][1],src[3][1],src[1][1],src[0][1]]
ax1.plot(x, y, color='#33cc99', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
ax1.set_ylim([h,0])
ax1.set_xlim([0,w])
ax1.set_title('Undistorted Image', fontsize=30)
ax2.imshow(exampleImg_unwarp)
ax2.set_title('Unwarped Image', fontsize=30)
plt.show()
'''
##############################################----Treshold---- ###################################################################

# Sobel x or y 를 정의한다.
# 그리고 absolute value를 취하고 threshold에 적용한다.
def abs_sobel_thresh(img, orient='x', thresh_min=25, thresh_max=255):
    # Apply the following steps to img
    # 1) LAB L channel 또는 grayscale로 변환한다.
    gray = (cv2.cvtColor(img, cv2.COLOR_RGB2Lab))[:, :, 0]
    # 2) 주어진 orient = 'x' or 'y'를 사용하여 x,y평면에서 미분한다.
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient == 'x', orient == 'y')
    # 3) 미분값(기울기)의 절대값을 취한다.
    abs_sobel = np.absolute(sobel)
    # 4) 8비트(0~255) scale 후 type = np.uint8 로 변환한다.
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) 그라데이션 크기가 thresh_min 보다 크고 thresh_max보다 작은 a mask of 1's 생성합니다.
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) mask를 binary_output image로 반환합니다.
    binary_output = sxbinary  # Remove this line
    return binary_output

def update(min_thresh, max_thresh):
    exampleImg_sobelAbs = abs_sobel_thresh(img_unwarp, 'x', min_thresh, max_thresh)
    # sobel absolute threshold 시각화한다.
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(img_unwarp)
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(exampleImg_sobelAbs, cmap='gray')
    ax2.set_title('Sobel Absolute', fontsize=30)

interact(update,min_thresh=(0, 255),max_thresh=(0, 255))


# Sobel x and y를 적용하고 기울기의 magnitude를 계산하여 thresold에 적용한다,
def mag_thresh(img, sobel_kernel=25, mag_thresh=(25, 255)):
    # Apply the following steps to img
    # 1) grayscale 변환한다.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) x,y 각각 기울기를 취한다.
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) magnitude를 계산한다.
    mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 4) 8비트(0~255) scale 후 type = np.uint8 로 변환한다.
    scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))
    # 5) magnitue thresold와 만나는 binary mask를 생성한다,
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) mask를 binary_output image로 반환합니다.
    binary_output = np.copy(sxbinary)
    return binary_output

def update(kernel_size, min_thresh, max_thresh):
    exampleImg_sobelMag = mag_thresh(img_unwarp, kernel_size, (min_thresh, max_thresh))
    # sobel magnitude threshold 를 시각화한다.
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(img_unwarp)
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(exampleImg_sobelMag, cmap='gray')
    ax2.set_title('Sobel Magnitude', fontsize=30)


interact(update, kernel_size=(1, 31, 2),
         min_thresh=(0, 255),
         max_thresh=(0, 255))


#  Sobel x and y를 적용하고 기울기의 방향을 계산하여 thresold에 적용한다.
def dir_thresh(img, sobel_kernel=7, thresh=(0, 0.09)):

    # 1) grayscale 변환한다.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) x,y 각각 기울기를 취한다.
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) 기울기의 절대값을 구한다.
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) 기울기의 방향을 계산한다.
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) direction thresholds에 해당하는 binarn mask를 생성한다.
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) mask를 binary_output image로 반환합니다.
    return binary_output


def update(kernel_size, min_thresh, max_thresh):
    exampleImg_sobelDir = dir_thresh(img_unwarp, kernel_size, (min_thresh, max_thresh))
    # sobel direction threshold 시각화한다.
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(img_unwarp)
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(exampleImg_sobelDir, cmap='gray')
    ax2.set_title('Sobel Direction', fontsize=30)


interact(update, kernel_size=(1, 31, 2),
         min_thresh=(0, np.pi / 2, 0.01),
         max_thresh=(0, np.pi / 2, 0.01))


def update(mag_kernel_size, mag_min_thresh, mag_max_thresh, dir_kernel_size, dir_min_thresh, dir_max_thresh):
    exampleImg_sobelMag2 = mag_thresh(img_unwarp, mag_kernel_size, (mag_min_thresh, mag_max_thresh))
    exampleImg_sobelDir2 = dir_thresh(img_unwarp, dir_kernel_size, (dir_min_thresh, dir_max_thresh))
    combined = np.zeros_like(exampleImg_sobelMag2)
    combined[((exampleImg_sobelMag2 == 1) & (exampleImg_sobelDir2 == 1))] = 1
    # sobel magnitude + direction threshold 2개를 혼합한 것을 시각화한다.
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(img_unwarp)
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Sobel Magnitude + Direction', fontsize=30)


interact(update, mag_kernel_size=(1, 31, 2),
         mag_min_thresh=(0, 255),
         mag_max_thresh=(0, 255),
         dir_kernel_size=(1, 31, 2),
         dir_min_thresh=(0, np.pi / 2, 0.01),
         dir_max_thresh=(0, np.pi / 2, 0.01))


# thresholds the S-channel of HLS 함수를 정의한다.
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_sthresh(img, thresh=(125, 255)):
    # 1) HLS color space로 변환한다.
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) S channel에 대한 thresold를 적용한다.
    binary_output = np.zeros_like(hls[:, :, 2])
    binary_output[(hls[:, :, 2] > thresh[0]) & (hls[:, :, 2] <= thresh[1])] = 1
    # 3) binary image of threshold result를 반환한다.
    return binary_output


def update(min_thresh, max_thresh):
    exampleImg_SThresh = hls_sthresh(img_unwarp, (min_thresh, max_thresh))
    # hls s-channel threshold 시각화한다.
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(img_unwarp)
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(exampleImg_SThresh, cmap='gray')
    ax2.set_title('HLS S-Channel', fontsize=30)


interact(update,
         min_thresh=(0, 255),
         max_thresh=(0, 255))
#thresholds the L-channel of HLS 함수를 정의한다.
def hls_lthresh(img, thresh=(220, 255)):
    # 1)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls_l = hls[:, :, 1]
    hls_l = hls_l * (255 / np.max(hls_l))
    # 2)
    binary_output = np.zeros_like(hls_l)
    binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
    # 3)
    return binary_output


def update(min_thresh, max_thresh):
    exampleImg_LThresh = hls_lthresh(img_unwarp, (min_thresh, max_thresh))
    # Visualize hls l-channel threshold
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(img_unwarp)
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(exampleImg_LThresh, cmap='gray')
    ax2.set_title('HLS L-Channel', fontsize=30)


interact(update,
         min_thresh=(0, 255),
         max_thresh=(0, 255))


# thresholds the B-channel of LAB 함수를 정의한다.
# Use exclusive lower bound (>) and inclusive upper (<=), OR the results of the thresholds (B channel should capture
# yellows)
def lab_bthresh(img, thresh=(190, 255)):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:, :, 2]
    # 이미지에 노란색이 없다면 정규화하지 않는다.
    if np.max(lab_b) > 175:
        lab_b = lab_b * (255 / np.max(lab_b))
    # 2)  L channel Thresold 를 적용한다(L이 맞다!!)
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3)
    return binary_output


def update(min_b_thresh, max_b_thresh):
    exampleImg_LBThresh = lab_bthresh(img_unwarp, (min_b_thresh, max_b_thresh))
    # Visualize LAB B threshold
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(img_unwarp)
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(exampleImg_LBThresh, cmap='gray')
    ax2.set_title('LAB B-channel', fontsize=30)


interact(update,
         min_b_thresh=(0, 255),
         max_b_thresh=(0, 255))
#----------------------------참고!!!!! 위 함수들은 적절한 임계값을 직접 시각적으로 조절할 수 있으나 쥬피터에서만 가동가능하다.----------------
#----------------------------쥬피터에서 적절한 값들을 찾아오자!!-------------------------------------------------

# pipline을 정의한다.(raw image 를 읽고 차선이 확인된 binary image를 반환한다.

def pipeline(img):
    # Undistort
    img_undistort = undistort(img)

    h, w = img_undistort.shape[:2]
    print(h)
    print(w)
    # define source and destination points for transform 변환할 포인트를 지정한다. figure 이미지 띄운 상태에서 값 찾아보자 dst의 경우w w변수로 유동적으로 작성하자.
    src = np.float32([(570, 460),
                      (700, 460),
                      (250, 680),
                      (1050, 680)])

    dst = np.float32([(400, 0),
                      (w - 400, 0),
                      (400, h),
                      (w - 400, h)]) #################################################### 1280,720 이미지에 대한 최적값이다. 구하느라 오래걸림;;;
###############################################################################################################카메라에 따라 다시 조정할것!!!!!!!!!!!!!!!!!!!!!!!!!##


    # Perspective Transform
    img_unwarp, M, Minv = unwarp(img_undistort, src, dst)

    # Sobel Absolute (using default parameters)
    # img_sobelAbs = abs_sobel_thresh(img_unwarp)

    # Sobel Magnitude (using default parameters)
    # img_sobelMag = mag_thresh(img_unwarp)

    # Sobel Direction (using default parameters)
    # img_sobelDir = dir_thresh(img_unwarp)

    # HLS S-channel Threshold (using default parameters)
    # img_SThresh = hls_sthresh(img_unwarp)

    # HLS L-channel Threshold (using default parameters)
    img_LThresh = hls_lthresh(img_unwarp)

    # Lab B-channel Threshold (using default parameters)
    img_BThresh = lab_bthresh(img_unwarp)

    # Combine HLS and Lab B channel thresholds
    combined = np.zeros_like(img_BThresh)
    combined[(img_LThresh == 1) | (img_BThresh == 1)] = 1


    return combined, Minv



#-----------------------------------sliding polyfit------------------------------------------

# sliding window를 이용하여 추출된 binary 라인 이미지에 polynomial을 fit하는 방법이다.
def sliding_window_polyfit(img):

    # 이미지(영상) 아래쪽 절반에 대한 히스토그램을 표시한다.
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    # 히스토그램에서 좌우 peak를 찾는다.
    # 그것이 좌 우 선의 스타팅 포인트이다.
    midpoint = np.int(histogram.shape[0] // 2)
    quarter_point = np.int(midpoint // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    # Peak in the second half indicates the likely position of the right lane
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    print(leftx_base, rightx_base)


    # Previously the left/right base was the max of the left/right half of the histogram
    # 이전의 히스토그램의 좌우 max값은 좌우 base값이였다.
    # 히스토그램의 1/4(왼쪽/오른쪽으로 기울임)만 고려하도록 변경합니다.

    ''' 아래 코드로 할 경우 x축 탐색 불가 이유는 모름 위 코드로 실행할 것 
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histogram[midpoint:(midpoint + quarter_point)]) + midpoint
    print(leftx_base, rightx_base)
    '''

    # print('base pts:', leftx_base, rightx_base)

    # 슬라이딩 창 갯수
    nwindows = 10
    # 윈도우 세로(heigt) 설정
    window_height = np.int(img.shape[0] / nwindows)
    # 이미지에서 0이 아닌 픽셀의 x,y포지션을 확인한다.
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # 각 창에서 현재 위치를 업데이트한다.
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # 최근 창에서 발견된 최소 픽셀 수를 설정한다.
    minpix = 40
    # 왼쪽 및 오른쪽 차선 픽셀 인덱스를 수신하는 빈 목록을 만든다.
    left_lane_inds = []
    right_lane_inds = []
    # 시각화를 위한 Rectangle data 목록을 만든다.
    rectangle_data = []

    # 창 하나 하나에 대한 단계이다.
    for window in range(nwindows):
        # 창의 경계영역을 확인한다.
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # 창 영역 내에서 0이 아닌 픽셀들을 확인한다.
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
        # 이 인덱스를 목록에 추가한다.(for문 위에 차선 픽셀 인덱스를 수신하는 빈 목록에 집어 넣는다)
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # minpix pixels보다 큰 값을 찾는다면, 새로운 창을 평균위치에 배치한다.
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # 인덱스 배열을 연결한다.
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 좌우 라인 픽셀 포지션을 추출한다.
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit, right_fit = (None, None)

    # 각각에 second order polynomial 피팅 시작한다.
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    visualization_data = (rectangle_data, histogram)

    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data#


# 이전의 적합한 값을 기반으로 다항식을 binary 이미지에 적용하는 방버이다. 반드시 이전의 과정이 선행되어야 한다.!!
# 이때 fit이 비디오의 한 프레임에서 다음 프레임으로 크게 변하지 않는다고 가정한다!!
def polyfit_using_prev_fit(binary_warped, left_fit_prev, right_fit_prev):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = ((nonzerox > (
                left_fit_prev[0] * (nonzeroy ** 2) + left_fit_prev[1] * nonzeroy + left_fit_prev[2] - margin)) &
                      (nonzerox < (left_fit_prev[0] * (nonzeroy ** 2) + left_fit_prev[1] * nonzeroy + left_fit_prev[
                          2] + margin)))
    right_lane_inds = ((nonzerox > (
                right_fit_prev[0] * (nonzeroy ** 2) + right_fit_prev[1] * nonzeroy + right_fit_prev[2] - margin)) &
                       (nonzerox < (right_fit_prev[0] * (nonzeroy ** 2) + right_fit_prev[1] * nonzeroy + right_fit_prev[
                           2] + margin)))

    # 다시한번 좌우 라인 픽셀 포지션을 추출한다.
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)
    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds

#----------------------------------------테스트 시작------------------------------------#

# Make a list of example images
images = glob.glob('./test image_line/*.jpg')

# Set up plot
fig, axs = plt.subplots(len(images), 2, figsize=(10, 20))
fig.subplots_adjust(hspace=.2, wspace=.001)
axs = axs.ravel()

i = 0

for image in images:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_bin, Minv = pipeline(img)
    print('...')
    print(img_bin)
    axs[i].imshow(img)
    axs[i].axis('off')
    i += 1
    axs[i].imshow(img_bin, cmap='gray')
    axs[i].axis('off')
    i += 1


#----------------------------------------테스트------------------------------------#

# visualize the result on example image
exampleImg = cv2.imread('./test image_line/solidWhiteRight.jpg')
exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)
exampleImg = cv2.resize(exampleImg, dsize=(1280,720), interpolation=cv2.INTER_AREA)


# pipline을 통해 thresold를 적용하고 perspective transformation 한 이미지를 받는다.
exampleImg_bin, Minv = pipeline(exampleImg)
plt.imshow(exampleImg_bin)
plt.show()

# sliding window 함수에 적용한다.
left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = sliding_window_polyfit(exampleImg_bin)

h = exampleImg.shape[0]
left_fit_x_int = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
right_fit_x_int = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]
# print('fit x-intercepts:', left_fit_x_int, right_fit_x_int)

rectangles = visualization_data[0]
histogram = visualization_data[1]


# 선을 그린 결과 이미지를 생성하고 결과를 시각화한다.
out_img = np.uint8(np.dstack((exampleImg_bin, exampleImg_bin, exampleImg_bin)) * 255)

# plotting을 위한 x,y값을 generate한다.
ploty = np.linspace(0, exampleImg_bin.shape[0] - 1, exampleImg_bin.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

for rect in rectangles:
    # 시각화 이미지에 창을 그린다.
    cv2.rectangle(out_img, (rect[2], rect[0]), (rect[3], rect[1]), (0, 255, 0), 2)
    cv2.rectangle(out_img, (rect[4], rect[0]), (rect[5], rect[1]), (0, 255, 0), 2)
# Identify the x and y positions of all nonzero pixels in the image
# 이미지에서 0이 아닌 픽셀값들의 x, y 포지션을 확인한다.
nonzero = exampleImg_bin.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100, 200, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)

# sliding window polyfit을 적용한 이미지로부터 히스토그램을 출력한다
plt.plot(histogram)
plt.xlim(0, 1280)
plt.show()
#----------------------------------------테스트------------------------------------


# visualize the result on example image
exampleImg2 = cv2.imread('./test image_line/solidWhiteCurve.jpg')
exampleImg2 = cv2.cvtColor(exampleImg2, cv2.COLOR_BGR2RGB)
exampleImg2 = cv2.resize(exampleImg2, dsize=(1280,720), interpolation=cv2.INTER_AREA)

exampleImg2_bin, Minv = pipeline(exampleImg2)
margin = 80

left_fit2, right_fit2, left_lane_inds2, right_lane_inds2 = polyfit_using_prev_fit(exampleImg2_bin, left_fit, right_fit)

# Generate x and y values for plotting
ploty = np.linspace(0, exampleImg2_bin.shape[0] - 1, exampleImg2_bin.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
left_fitx2 = left_fit2[0] * ploty ** 2 + left_fit2[1] * ploty + left_fit2[2]
right_fitx2 = right_fit2[0] * ploty ** 2 + right_fit2[1] * ploty + right_fit2[2]

# 이미지에 선을 그리고 선택 창 영역을 보여준다.
out_img = np.uint8(np.dstack((exampleImg2_bin, exampleImg2_bin, exampleImg2_bin)) * 255)
window_img = np.zeros_like(out_img)

# 좌우 라인 픽셀에 색깔 입히기
nonzero = exampleImg2_bin.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
out_img[nonzeroy[left_lane_inds2], nonzerox[left_lane_inds2]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds2], nonzerox[right_lane_inds2]] = [0, 0, 255]

# polygon을 생성하여 검색 창 영역(OLD FIT)을 보여 주고 x 및 y 점을 cv2fillPoly()에 사용할 수 있는 형식으로 재구성합니다.
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# wraped 이미지에 선을 그린다.
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.imshow(result)
plt.plot(left_fitx2, ploty, color='yellow')
plt.plot(right_fitx2, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()
#----------------------------------------테스트 종료------------------------------------


#----------------------------------------곡률 반경 계산 및 적용하기----------------------------------

# Method to determine radius of curvature and distance from lane center
# based on binary image, polynomial fit, and L and R lane pixel indices
def calc_curv_rad_and_center_dist(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3.048 / 100  # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7 / 378  # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    left_curverad, right_curverad, center_dist = (0, 0, 0)
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = bin_img.shape[0]
    ploty = np.linspace(0, h - 1, h)
    y_eval = np.max(ploty)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds]
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]

    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                    2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters

    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts
    if r_fit is not None and l_fit is not None:
        car_position = bin_img.shape[1] / 2
        l_fit_x_int = l_fit[0] * h ** 2 + l_fit[1] * h + l_fit[2]
        r_fit_x_int = r_fit[0] * h ** 2 + r_fit[1] * h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center_dist = (car_position - lane_center_position) * xm_per_pix
    return left_curverad, right_curverad, center_dist

rad_l, rad_r, d_center = calc_curv_rad_and_center_dist(exampleImg_bin, left_fit, right_fit, left_lane_inds,
                                                       right_lane_inds)

print('Radius of curvature for example:', rad_l, 'm,', rad_r, 'm')
print('Distance from lane center for example:', d_center, 'm')

#----------------------원본 이미지에 라인 그리기--------------------------------------------------------------

def draw_lane(original_img, binary_img, l_fit, r_fit, Minv):
    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h, w = binary_img.shape
    ploty = np.linspace(0, h - 1, num=h)  # to cover same y-range as image
    left_fitx = l_fit[0] * ploty ** 2 + l_fit[1] * ploty + l_fit[2]
    right_fitx = r_fit[0] * ploty ** 2 + r_fit[1] * ploty + r_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 255, 255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result

exampleImg_out1 = draw_lane(exampleImg, exampleImg_bin, left_fit, right_fit, Minv)
plt.imshow(exampleImg_out1)
plt.show()



def draw_data(original_img, curv_rad, center_dist):
    new_img = np.copy(original_img)
    h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(new_img, text, (40, 70), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40, 120), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)
    return new_img

exampleImg_out2 = draw_data(exampleImg_out1, (rad_l + rad_r) / 2, d_center)
plt.imshow(exampleImg_out2)
plt.show()


#----------------------데이터 저장을 위한 Line Class 정의--------------------------------------------------------------

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # number of detected pixels
        self.px_count = None

    def add_fit(self, fit, inds):
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit - self.best_fit)
            if (self.diffs[0] > 0.001 or \
                self.diffs[1] > 1.0 or \
                self.diffs[2] > 100.) and \
                    len(self.current_fit) > 0:
                # bad fit! abort! abort! ... well, unless there are no fits in the current_fit queue, then we'll take it
                self.detected = False
            else:
                self.detected = True
                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit) - 5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # or remove one from the history, if not found
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit) - 1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)



#------------------------------------final pipline------------------------------

def process_image(img):
    new_img = np.copy(img)
    img_bin, Minv = pipeline(new_img)

    # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
    if not l_line.detected or not r_line.detected:
        l_fit, r_fit, l_lane_inds, r_lane_inds, _ = sliding_window_polyfit(img_bin)
    else:
        l_fit, r_fit, l_lane_inds, r_lane_inds = polyfit_using_prev_fit(img_bin, l_line.best_fit, r_line.best_fit)

    # invalidate both fits if the difference in their x-intercepts isn't around 350 px (+/- 100 px)
    if l_fit is not None and r_fit is not None:
        # calculate x-intercept (bottom of image, x=image_height) for fits
        h = img.shape[0]
        l_fit_x_int = l_fit[0] * h ** 2 + l_fit[1] * h + l_fit[2]
        r_fit_x_int = r_fit[0] * h ** 2 + r_fit[1] * h + r_fit[2]
        x_int_diff = abs(r_fit_x_int - l_fit_x_int)
        if abs(350 - x_int_diff) > 100:
            l_fit = None
            r_fit = None

    l_line.add_fit(l_fit, l_lane_inds)
    r_line.add_fit(r_fit, r_lane_inds)

    # draw the current best fit if it exists
    if l_line.best_fit is not None and r_line.best_fit is not None:
        img_out1 = draw_lane(new_img, img_bin, l_line.best_fit, r_line.best_fit, Minv)
        rad_l, rad_r, d_center = calc_curv_rad_and_center_dist(img_bin, l_line.best_fit, r_line.best_fit,
                                                               l_lane_inds, r_lane_inds)
        img_out = draw_data(img_out1, (rad_l + rad_r) / 2, d_center)
    else:
        img_out = new_img

    diagnostic_output = False
    if diagnostic_output:
        # put together multi-view output
        diag_img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # original output (top left)
        diag_img[0:360, 0:640, :] = cv2.resize(img_out, (640, 360))

        # binary overhead view (top right)
        img_bin = np.dstack((img_bin * 255, img_bin * 255, img_bin * 255))
        resized_img_bin = cv2.resize(img_bin, (640, 360))
        diag_img[0:360, 640:1280, :] = resized_img_bin

        # overhead with all fits added (bottom right)
        img_bin_fit = np.copy(img_bin)
        for i, fit in enumerate(l_line.current_fit):
            img_bin_fit = plot_fit_onto_img(img_bin_fit, fit, (20 * i + 100, 0, 20 * i + 100))
        for i, fit in enumerate(r_line.current_fit):
            img_bin_fit = plot_fit_onto_img(img_bin_fit, fit, (0, 20 * i + 100, 20 * i + 100))
        img_bin_fit = plot_fit_onto_img(img_bin_fit, l_line.best_fit, (255, 255, 0))
        img_bin_fit = plot_fit_onto_img(img_bin_fit, r_line.best_fit, (255, 255, 0))
        diag_img[360:720, 640:1280, :] = cv2.resize(img_bin_fit, (640, 360))

        # diagnostic data (bottom left)
        color_ok = (200, 255, 155)
        color_bad = (255, 155, 155)
        font = cv2.FONT_HERSHEY_DUPLEX
        if l_fit is not None:
            text = 'This fit L: ' + ' {:0.6f}'.format(l_fit[0]) + \
                   ' {:0.6f}'.format(l_fit[1]) + \
                   ' {:0.6f}'.format(l_fit[2])
        else:
            text = 'This fit L: None'
        cv2.putText(diag_img, text, (40, 380), font, .5, color_ok, 1, cv2.LINE_AA)
        if r_fit is not None:
            text = 'This fit R: ' + ' {:0.6f}'.format(r_fit[0]) + \
                   ' {:0.6f}'.format(r_fit[1]) + \
                   ' {:0.6f}'.format(r_fit[2])
        else:
            text = 'This fit R: None'
        cv2.putText(diag_img, text, (40, 400), font, .5, color_ok, 1, cv2.LINE_AA)
        text = 'Best fit L: ' + ' {:0.6f}'.format(l_line.best_fit[0]) + \
               ' {:0.6f}'.format(l_line.best_fit[1]) + \
               ' {:0.6f}'.format(l_line.best_fit[2])
        cv2.putText(diag_img, text, (40, 440), font, .5, color_ok, 1, cv2.LINE_AA)
        text = 'Best fit R: ' + ' {:0.6f}'.format(r_line.best_fit[0]) + \
               ' {:0.6f}'.format(r_line.best_fit[1]) + \
               ' {:0.6f}'.format(r_line.best_fit[2])
        cv2.putText(diag_img, text, (40, 460), font, .5, color_ok, 1, cv2.LINE_AA)
        text = 'Diffs L: ' + ' {:0.6f}'.format(l_line.diffs[0]) + \
               ' {:0.6f}'.format(l_line.diffs[1]) + \
               ' {:0.6f}'.format(l_line.diffs[2])
        if l_line.diffs[0] > 0.001 or \
                l_line.diffs[1] > 1.0 or \
                l_line.diffs[2] > 100.:
            diffs_color = color_bad
        else:
            diffs_color = color_ok
        cv2.putText(diag_img, text, (40, 500), font, .5, diffs_color, 1, cv2.LINE_AA)
        text = 'Diffs R: ' + ' {:0.6f}'.format(r_line.diffs[0]) + \
               ' {:0.6f}'.format(r_line.diffs[1]) + \
               ' {:0.6f}'.format(r_line.diffs[2])
        if r_line.diffs[0] > 0.001 or \
                r_line.diffs[1] > 1.0 or \
                r_line.diffs[2] > 100.:
            diffs_color = color_bad
        else:
            diffs_color = color_ok
        cv2.putText(diag_img, text, (40, 520), font, .5, diffs_color, 1, cv2.LINE_AA)
        text = 'Good fit count L:' + str(len(l_line.current_fit))
        cv2.putText(diag_img, text, (40, 560), font, .5, color_ok, 1, cv2.LINE_AA)
        text = 'Good fit count R:' + str(len(r_line.current_fit))
        cv2.putText(diag_img, text, (40, 580), font, .5, color_ok, 1, cv2.LINE_AA)

        img_out = diag_img
    return img_out


def plot_fit_onto_img(img, fit, plot_color):
    if fit is None:
        return img
    new_img = np.copy(img)
    h = new_img.shape[0]
    ploty = np.linspace(0, h - 1, h)
    plotx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
    pts = np.array([np.transpose(np.vstack([plotx, ploty]))])
    cv2.polylines(new_img, np.int32([pts]), isClosed=False, color=plot_color, thickness=8)
    return new_img

#---------------------------------비디오 실행---------------------------------------
'''
l_line = Line()
r_line = Line()
#my_clip.write_gif('test.gif', fps=12)
video_output1 = 'project_video_output.mp4'
video_input1 = VideoFileClip('UK_vod.mp4')#.subclip(22,26)
processed_video = video_input1.fl_image(process_image)
'''

input_name = 'challenge_video.mp4'
l_line = Line()
r_line = Line()
cap = cv2.VideoCapture(input_name)
while (cap.isOpened()):
    _, frame = cap.read()

    resized_img = cv2.resize(frame, dsize=(1280,720), interpolation=cv2.INTER_AREA)


    # resized_img = np.array(frame.resize((1280, 720,3)))  # width, height
    result = process_image(resized_img)
    cv2.imshow('result', result.astype(np.uint8))
    # out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
       cv2.waitKey(0)
        #if cv2.waitKey(1) & 0xFF == ord('r'):
        #    cv2.imwrite('check1.jpg', undist_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


''''
capture = cv2.VideoCapture('project_video.mp4')

while capture.isOpened():
    run, frame = capture.read()
    if not run:
        print("[프레임 수신 불가] - 종료합니다")
        break
    img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    cv2.imshow('video', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
'''

'''
# Correcting for Distortion
undist_img = undistort(frame, mtx, dist)
# resize video
undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
rows, cols = undist_img.shape[:2]

combined_gradient = gradient_combine(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
#cv2.imshow('gradient combined image', combined_gradient)

combined_hls = hls_combine(undist_img, th_h, th_l, th_s)
#cv2.imshow('HLS combined image', combined_hls)

combined_result = comb_result(combined_gradient, combined_hls)

c_rows, c_cols = combined_result.shape[:2]
s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

warp_img, M, Minv = warp_image(combined_result, src, dst, (720, 720))
#cv2.imshow('warp', warp_img)

searching_img = find_LR_lines(warp_img, left_line, right_line)
#cv2.imshow('LR searching', searching_img)

w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line)
#cv2.imshow('w_comb_result', w_comb_result)

# Drawing the lines back down onto the road
color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
lane_color = np.zeros_like(undist_img)
lane_color[220:rows - 12, 0:cols] = color_result

# Combine the result with the original image
result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)
#cv2.imshow('result', result.astype(np.uint8))

info, info2 = np.zeros_like(result),  np.zeros_like(result)
info[5:110, 5:190] = (255, 255, 255)
info2[5:110, cols-111:cols-6] = (255, 255, 255)
info = cv2.addWeighted(result, 1, info, 0.2, 0)
info2 = cv2.addWeighted(info, 1, info2, 0.2, 0)
road_map = print_road_map(w_color_result, left_line, right_line)
info2[10:105, cols-106:cols-11] = road_map
info2 = print_road_status(info2, left_line, right_line)
'''






























