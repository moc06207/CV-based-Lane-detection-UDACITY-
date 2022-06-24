#-*-coding:utf-8-*-
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
from Sliding_radian import sliding_window_polyfit,polyfit_using_prev_fit,draw_lane,draw_data,calc_curv_rad_and_center_dist
from thresold import abs_sobel_thresh,mag_thresh,dir_thresh,hls_sthresh, hls_lthresh ,lab_bthresh,rgb_rthresh,hls_sthresh
# from undistotr_unwarped import undistort,unwarp

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real worldls space
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
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)


# undistort image using camera calibration matrix from above
def undistort(img):
    img = cv2.undistort(img, mtx, dist, None, mtx)
    return img


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


# pipline을 정의한다.(raw image 를 읽고 차선이 확인된 binary image를 반환한다.

def pipeline(img):
    # Undistort
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_undistort = undistort(img)

    # plt.imshow(img_undistort)
    # plt.show()
    h, w = img_undistort.shape[:2]

    # define source and destination points for transform 변환할 포인트를 지정한다. figure 이미지 띄운 상태에서 값 찾아보자 dst의 경우w w변수로 유동적으로 작성하자.
    # k-city camera-3 전용
    # src = np.float32([(353, 224),
    #                   (700, 224),
    #                   (30, 600),
    #                   (990, 600)])
    #
    # dst = np.float32([(570, 50),
    #                   (w - 570, 50),
    #                   (570, h),
    #                   (w - 570, h)])

    # # ---밑은 유닥시티 전용
    src = np.float32([(575, 464),
                      (707, 464),
                      (258, 682),
                      (1049, 682)])
    dst = np.float32([(450, 0),
                      (w - 450, 0),
                      (450, h),
                      (w - 450, h)])



    # dst = np.float32([(400, 0),
    #                   (w - 400, 0),
    #                   (400, h),
    #                   (w - 400, h)]) #################################################### 1280,720 이미지에 대한 최적값이다. 구하느라 오래걸림;;;
###############################################################################################################카메라에 따라 다시 조정할것!!!!!!!!!!!!!!!!!!!!!!!!!##
    # Perspective Transform
    img_unwarp, M, Minv = unwarp(img_undistort, src, dst)
    ###############################################COLOR SPACE################################################q
    img_unwarp_R = img_unwarp[:, :, 0]
    img_unwarp_G = img_unwarp[:, :, 1]
    img_unwarp_B = img_unwarp[:, :, 2]
    img_unwarp_HSV = cv2.cvtColor(img_unwarp, cv2.COLOR_RGB2HSV)
    img_unwarp_H = img_unwarp_HSV[:, :, 0]
    img_unwarp_S = img_unwarp_HSV[:, :, 1]
    img_unwarp_V = img_unwarp_HSV[:, :, 2]
    img_unwarp_LAB = cv2.cvtColor(img_unwarp, cv2.COLOR_RGB2Lab)
    img_unwarp_L = img_unwarp_LAB[:, :, 0]
    img_unwarp_A = img_unwarp_LAB[:, :, 1]
    img_unwarp_B2 = img_unwarp_LAB[:, :, 2]


    # Sobel Absolute (using default parameters)
    # img_sobelAbs = abs_sobel_thresh(img_unwarp)

    # Sobel Magnitude (using default parameters)
    # img_sobelMag = mag_thresh(img_unwarp)

    # Sobel Direction (using default parameters)
    # img_sobelDir = dir_thresh(img_unwarp)

    # RGB R-channel
    img_RThresh = rgb_rthresh(img_unwarp)

    # HLS S-channel Threshold (using default parameters)
    img_SThresh = hls_sthresh(img_unwarp)

    # HLS L-channel Threshold (using default parameters)
    #img_LThresh = hls_lthresh(img_unwarp)

    # Lab B-channel Threshold (using default parameters)
    #img_BThresh = lab_bthresh(img_unwarp)

    # Combine HLS and Lab B channel thresholds
    combined = np.zeros_like(img_SThresh)
    combined[(img_RThresh == 1) | (img_SThresh == 1)] = 1
    print("...")
    print(combined)
    return combined, Minv, img_unwarp,img_unwarp_R,img_unwarp_G, img_unwarp_B,img_unwarp_H,img_unwarp_S,img_unwarp_V,img_unwarp_L, img_unwarp_A,img_unwarp_B2
#----------------------데이터 저장을 위한 Line Class 정의--------------------------------------------------------------

# Define a class to receive the characteristics of each line detection
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
        self.current_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #number of detected pixels
        self.px_count = None
        # starting x_value
        self.startx = None
        # ending x_value
        self.endx = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # road information
        self.road_inf = None
        self.curvature = None
        self.deviation = None
        # Set the width of the windows +/- margin
        self.window_margin = 60
        # x values of the fitted line over the last n iterations
        self.prevx = []

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


#------------------------------------final pipline---------------------------------------------------

def process_image(img):
    new_img = np.copy(img)

    img_bin, Minv, img_unwarp, img_unwarp_R,img_unwarp_G, img_unwarp_B,img_unwarp_H,img_unwarp_S,img_unwarp_V,img_unwarp_L, img_unwarp_A,img_unwarp_B2= pipeline(new_img)

    global visualization_data

    thresold_image  = img_bin
    bird_view = img_unwarp

    # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
    if not l_line.detected or not r_line.detected:
        l_fit, r_fit, l_lane_inds, r_lane_inds, visualization_data = sliding_window_polyfit(img_bin)
    else:
        l_fit, r_fit, l_lane_inds, r_lane_inds = polyfit_using_prev_fit(img_bin, l_line.best_fit, r_line.best_fit)
#-------------------------------------------밑에는 히스토그램--------------------------------------------------
    #h = img.shape[0]
    # left_fit_x_int = l_fit[0] * h ** 2 + l_fit[1] * h + l_fit[2]
    # right_fit_x_int = r_fit[0] * h ** 2 + r_fit[1] * h + r_fit[2]
    rectangles = visualization_data[0]
    #histogram = visualization_data[1]
    # 선을 그린 결과 이미지를 생성하고 결과를 시각화한다
    out_img = np.uint8(np.dstack((img_bin, img_bin, img_bin)) * 255)

    # plotting을 위한 x,y값을 generate한다.
    ploty = np.linspace(0, img_bin.shape[0] - 1, img_bin.shape[0])
    # left_fitx = l_fit[0] * ploty ** 2 + l_fit[1] * ploty + l_fit[2]
    # right_fitx = r_fit[0] * ploty ** 2 + r_fit[1] * ploty + r_fit[2]

    for rect in rectangles:
        # 시각화 이미지에 창을 그린다.
        cv2.rectangle(out_img, (rect[2], rect[0]), (rect[3], rect[1]), (0, 255, 0), 2)
        cv2.rectangle(out_img, (rect[4], rect[0]), (rect[5], rect[1]), (0, 255, 0), 2)
    # Identify the x and y positions of all nonzero pixels in the image
    # 이미지에서 0이 아닌 픽셀값들의 x, y 포지션을 확인한다.
    nonzero = img_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[l_lane_inds], nonzerox[l_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[r_lane_inds], nonzerox[r_lane_inds]] = [100, 200, 255]
    # -------------------------------------------위에는 히스토그램--------------------------------------------------

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
        img_out1 = draw_lane(new_img, img_bin, l_line.best_fit, r_line.best_fit, Minv)######################
        rad_l, rad_r, d_center = calc_curv_rad_and_center_dist(img_bin, l_line.best_fit, r_line.best_fit,
                                                               l_lane_inds, r_lane_inds)
        img_out = draw_data(img_out1, (rad_l + rad_r) / 2, d_center)
    else:
        img_out = new_img

    line_draw_image = img_out

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

    return img_out, line_draw_image, bird_view,thresold_image,out_img, img_unwarp_R,img_unwarp_G, img_unwarp_B,img_unwarp_H,img_unwarp_S,img_unwarp_V,img_unwarp_L, img_unwarp_A,img_unwarp_B2


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

input_name = 'project_video.mp4'
#project_video.mp4
#kcity_camera_3.avi
l_line = Line()
r_line = Line()
cap = cv2.VideoCapture(input_name)
while (cap.isOpened()):
    _, frame = cap.read()

    undist_img = undistort(frame)


    undist_img = cv2.resize(undist_img,dsize=(1280,720), interpolation=cv2.INTER_AREA)


    # resized_img = np.array(frame.resize((1280, 720,3)))  # width, height #Process image
    result,line_draw_image, bird_view ,thresold_image, bird_view_line, img_unwarp_R,img_unwarp_G, img_unwarp_B,img_unwarp_H,img_unwarp_S,img_unwarp_V,img_unwarp_L, img_unwarp_A,img_unwarp_B2 = process_image(undist_img)

   #  cv2.imshow('result', result.astype(np.uint8))

    h, w = result.shape[:2]


    # color space
    img_unwarp_R = cv2.resize(img_unwarp_R, (16 * 25, 9 * 25))
    img_unwarp_G = cv2.resize(img_unwarp_G, (16 * 25, 9 * 25))
    img_unwarp_B = cv2.resize(img_unwarp_B, (16 * 25, 9 * 25))
    img_unwarp_H = cv2.resize(img_unwarp_H, (16 * 25, 9 * 25))
    img_unwarp_L = cv2.resize(img_unwarp_S, (16 * 25, 9 * 25))
    img_unwarp_S = cv2.resize(img_unwarp_V, (16 * 25, 9 * 25))
    img_unwarp_L = cv2.resize(img_unwarp_L, (16 * 25, 9 * 25))
    img_unwarp_A = cv2.resize(img_unwarp_A, (16 * 25, 9 * 25))
    img_unwarp_B2 = cv2.resize(img_unwarp_B2, (16 * 25, 9 * 25))

    add_Color_h1 = np.hstack([img_unwarp_R, img_unwarp_G,img_unwarp_B])
    add_Color_h2 = np.hstack([img_unwarp_H, img_unwarp_L, img_unwarp_S])
    add_Color_h3 = np.hstack([img_unwarp_L, img_unwarp_A, img_unwarp_B2])

    # 16*80 =1280, 9*80=720
    thresold_image = cv2.resize(thresold_image, (16*50, 9*50))
    bird_view_line = cv2.resize(bird_view_line, (16 * 50, 9 * 50))

    bird_view = cv2.resize(bird_view, (16*30, 9*50))
    result = cv2.resize(result, (16*50, 9*50))
    line_draw_image = cv2.resize(line_draw_image, (16 * 30, 9 * 50))

    addh1 = np.hstack([result.astype(np.uint8),bird_view])
    addh2 = np.hstack([addh1,line_draw_image])


    cv2.imshow("RGB", add_Color_h1)
    cv2.imshow("HLS", add_Color_h2)
    cv2.imshow("LAB", add_Color_h3)

    # cv2.imshow("img_unwarp_R", img_unwarp_R)
    # cv2.imshow("img_unwarp_G", img_unwarp_G)
    # cv2.imshow("img_unwarp_B", img_unwarp_B)
    # cv2.imshow("img_unwarp_H", img_unwarp_H)
    # cv2.imshow("img_unwarp_S", img_unwarp_S)
    # cv2.imshow("img_unwarp_V", img_unwarp_V)
    # cv2.imshow("img_unwarp_L", img_unwarp_L)
    # cv2.imshow("img_unwarp_A", img_unwarp_A)
    # cv2.imshow("img_unwarp_B2", img_unwarp_B2)




    cv2.imshow("bird_view_line", bird_view_line)
    cv2.imshow('thresold_image', thresold_image)
    cv2.imshow('result',addh2)

    #cv2.imshow('bird_view', bird_view)
    # cv2.imshow('result', result.astype(np.uint8))
    # out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('s'): # s 누르면 영상 재생됨
       cv2.waitKey(0)
        #if cv2.waitKey(1) & 0xFF == ord('r'):
        #    cv2.imwrite('check1.jpg', undist_img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # q누르면 영상 종료
        break

cap.release()
cv2.destroyAllWindows()