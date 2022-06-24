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
    # print(leftx_base, rightx_base)


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

    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data


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


def calc_curv_rad_and_center_dist(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3.048 / 100  # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.5 / 378  # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
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


#----------------------원본 이미지에 라인 그리기--------------------------------------------------------------

def draw_lane(original_img, binary_img, l_fit, r_fit, Minv):


    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img,
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
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 255), thickness=8)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 255, 255), thickness=8)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result


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