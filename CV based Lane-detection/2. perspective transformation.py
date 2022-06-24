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


# 이미지에 undistortion 테스트
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

def unwarp(img, src, dst):

    h, w = img.shape[:2]  # 밑에 w 인식을 못해서 다시 씀
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() wrap이미지를 top-town 관점에서 실행한다.
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    plt.imshow(warped)
    return warped, M, Minv

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


def pipeline(img):
    # Undistort
    img_undistort = undistort(img)

    h, w = img_undistort.shape[:2]
    print(h)
    print(w)
    # define source and destination points for transform 변환할 포인트를 지정한다. figure 이미지 띄운 상태에서 값 찾아보자 dst의 경우w w변수로 유동적으로 작성하자.
    src = np.float32([(0.3 * w, 0.4 * h),
                      (0.5 * w, 0.4 * h),
                      (0.2 * w, h),
                      (0.8 * w, h)])
    dst = np.float32([(250, 0),
                      (w - 400, 0),
                      (250, h),
                      (w - 400, h)])

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


#----------------------------------------테스트------------------------------------#

# visualize the result on example image
img_unwarp = cv2.imread('./test image_line/solidWhiteRight.jpg')
img_unwarp = cv2.cvtColor(img_unwarp, cv2.COLOR_BGR2RGB)

print(img_unwarp.shape)

# pipline을 통해 thresold를 적용하고 perspective transformation 한 이미지를 받는다.
exampleImg_bin, Minv = pipeline(img_unwarp)
plt.imshow(exampleImg_bin)
plt.show()

