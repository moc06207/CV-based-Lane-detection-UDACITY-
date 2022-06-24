import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from PIL import Image
import matplotlib.image as mpimg

# undistort image using camera calibration matrix from above
def undistort(new_img):
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

    cv2.imread('./calibration image/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    # 위에서 실행한 camera calibration 값들을 다른 변수에 저장하고 실행한다./cv2이용한다.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # 나중에 사용하기 위해 calibration값을 저장해둔다. (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("calibration.p", "wb"))
    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    return cv2.undistort(new_img, mtx, dist, None, mtx)

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