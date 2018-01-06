# -*- coding: utf-8 -*-
# python 3.6.1
import os
import time
import math
import random

import numpy as np
import cv2

# 保存截屏的文件位置
SCREENSHOT_FILE_NAME = 'screenshot.png'

# 按压因子
PRESS_FACTOR = 1.393

# 当游戏结束时，图片亮度的阀值
GAME_OVER_BRIGHTNESS_THRESH = 150

# 棋子球状头部的面积与周长参数
HEAD_AREA = 2919.10
HEAD_AREA_DELTA = 20
HEAD_PERIMETER = 201.90
HEAD_PERIMETER_DELTA = 4

# 空路径
NONE_PATH = -1, -1, -1, -1


def pull_screenshot(out_file):
    """
    通过adb截屏，并将截屏图片保存到当前目录。

    :param out_file: 保存截屏的文件名
    :return:
    """
    os.system('adb shell screencap -p /sdcard/screen.png')
    os.system(
        'adb pull /sdcard/screen.png {} > /dev/null 2>&1'.format(out_file))


def jump(x1, y1, x2, y2):
    """
    通过adb，让棋子从(x1,y1)跳到(x2,y2)。

    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    pressing_time = math.hypot(x1 - x2, y1 - y2) * PRESS_FACTOR
    os.system('adb shell input swipe {} {} {} {} {}'.format(
        x1, y1, x2, y2, int(pressing_time)))


def is_all_position_in_range(contour, height):
    """
    判断contour中的点是否都在指定范围内。

    棋子的球状头部一般只存在于图片的中间位置，为了避免干扰，
    将图片的上面一部分以及下面一部分给排除。

    :param contour:
    :param height: contour所在的图片的高度
    :return:
    """
    for pos in contour:
        _, y = pos[0]
        if not height / 6 < y < height / 6 * 5:
            return False
    return True


def is_chess_head(contour):
    """
    判断给出的contour是否为棋子的球状头部。

    :param contour:
    :return:
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # perimeter mean = 201.90, max = 203.82, min = 200.99
    # area mean = 2919.10, max = 2928.5, min = 2910.0
    return math.fabs(area - HEAD_AREA) < HEAD_AREA_DELTA and \
           math.fabs(perimeter - HEAD_PERIMETER) < HEAD_PERIMETER_DELTA


def find_contours(grey_img):
    """
    在图片grey_img中寻找并返回contours。

    :param grey_img:
    :return:
    """
    # 因为在图片不同的区域明度会有不同，
    # 所有这里使用Adaptive Thresholding将图片的各个色块的轮廓给找出来。
    thresh_img = cv2.adaptiveThreshold(grey_img, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 1)
    _, contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect_chess_head(img):
    """
    在图片中找出棋子头部对应的contour，如果没找到返回一个空数组。

    :param img:
    :return:
    """
    height, _, _ = img.shape

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours = find_contours(grey_img)
    legal_contours = [c for c in contours if
                      is_all_position_in_range(c, height) and
                      is_chess_head(c)]

    if len(legal_contours) == 1:
        return legal_contours[0]
    else:
        return np.zeros((0, 0))


def detect_chess_head_special(img):
    """
    当棋子球状头部与其他区域有重叠时，使用一般的方法不好能找出它，
    这里先用棋子对应的颜色区间将棋子给过滤出来，然后再放宽面积与周长的条件，
    查找出它的位置。

    :param img:
    :return:
    """
    height, _, _ = img.shape
    lower = np.array([63, 43, 41])
    upper = np.array([149, 102, 113])
    grey_img = cv2.inRange(img, lower, upper)
    contours = find_contours(grey_img)
    legal_contours = [c for c in contours if
                      is_all_position_in_range(c, height) and
                      2400 < cv2.contourArea(c) < 3000 and
                      190 < cv2.arcLength(c, True) < 250]
    if len(legal_contours) == 1:
        return legal_contours[0]
    else:
        return np.zeros((0, 0))


def crop_image(img, chess_head):
    """
    根据棋子球状头部的位置，裁剪出目标大概对应的区域，
    减少下一步处理图片时的干扰。

    :param img:
    :param chess_head:
    :return:
    """
    img = img.copy()
    height, width, _ = img.shape
    perimeter = cv2.arcLength(chess_head, True)
    moments = cv2.moments(chess_head)
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']

    img[:height // 6, :, :] = (0, 0, 0)
    img[int(cy + perimeter * 0.9):, :, :] = (0, 0, 0)
    if cx < width / 2:
        # 棋子在图片的左边
        img[:, :int(cx + perimeter / math.pi), :] = (0, 0, 0)
    else:
        # 棋子在图片的右边
        img[:, int(cx - perimeter / math.pi):, :] = (0, 0, 0)
    return img


def detect_target_bound(img, chess_head):
    """
    探测目标区域的大概边界。

    :param img:
    :param chess_head:
    :return:
    """
    img = img.copy()

    height, width, _ = img.shape

    perimeter = cv2.arcLength(chess_head, True)

    moments = cv2.moments(chess_head)
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']

    if cx < width / 2:
        # left
        return (int(cx + perimeter / math.pi), height // 6,
                width, int(cy + perimeter * 0.9))
    else:
        # right
        return (0, height // 6,
                int(cx - perimeter / math.pi), int(cy + perimeter * 0.9))


def detect_target_position(img, chess_head):
    """
    探测目标的位置。

    :param img:
    :param chess_head:
    :return:
    """
    x1, y1, x2, y2 = detect_target_bound(img, chess_head)

    target_img = np.zeros(shape=img.shape, dtype=np.uint8)
    target_img[y1:y2, x1:x2, :] = img[y1:y2, x1:x2, :]

    # BGR2GRAY与RGB2GRAY两种contours都考虑进去，可以避免一些浅色的物体无法被识别的问题
    contours = find_contours(cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY))
    contours += find_contours(cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY))
    contours = [c for c in contours if 100 < cv2.contourArea(c)]

    points = []
    for c in contours:
        for p in c:
            x, y = p[0]
            if y1 + 10 < y < y2 - 10 and x1 + 10 < x < x2 - 10:
                points.append((x, y))
    points.sort(key=lambda p: p[1])

    top_point = points[0]
    return top_point


def detect_path(img):
    """
    探测路径，也就是两个点，起点与终点。

    :param img:
    :return:
    """
    _, width, _ = img.shape

    chess_head = detect_chess_head(img)
    if chess_head.size == 0:
        chess_head = detect_chess_head_special(img)

    if chess_head.size == 0:
        print("Can't find chess head")
        return -1, -1, -1, -1

    perimeter = cv2.arcLength(chess_head, True)
    moments = cv2.moments(chess_head)

    # 中心点(x, y)
    start_x = moments['m10'] / moments['m00']
    start_y = moments['m01'] / moments['m00']
    start_y += perimeter * 0.8

    end_x, end_y = detect_target_position(img, chess_head)
    # 当跳动时，有很多情况够不着的情况，所以这里把目标的距离再加上10
    if end_x < width / 2:
        end_x -= 10
    else:
        end_x += 10

    return int(start_x), int(start_y), int(end_x), int(end_y)


def is_game_over(img):
    """
    判断是否游戏结束了。

    :param img:
    :return:
    """
    height, width, _ = img.shape

    # 当游戏结束时，图片的整体亮度会降低，根据这个来判断
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lightness = 0
    for row in hsv[:, :, 2]:
        for value in row:
            lightness += value
    lightness /= height * width

    return lightness < 150


def main():
    i = 0
    game_over = False

    while not game_over:
        i += 1

        # 截屏
        pull_screenshot(SCREENSHOT_FILE_NAME)

        # 读取图片
        img = cv2.imread(SCREENSHOT_FILE_NAME)

        if not is_game_over(img):
            height, width, _ = img.shape

            # 探测起点与终点
            x1, y1, x2, y2 = detect_path(img)
            print('step {:04}: ({:04}, {:04}) -> ({:04}, {:04})'.format(
                i, x1, y1, x2, y2))

            jump(x1, y1, x2, y2)

            # 标记起点与终点
            cv2.circle(img, (x1, y1), 5, (0, 255, 0), -1)
            cv2.circle(img, (x2, y2), 5, (0, 255, 0), -1)
            cv2.imwrite(SCREENSHOT_FILE_NAME, img)

            time.sleep(random.uniform(1.8, 2.2))
        else:
            print('game over')
            game_over = True


if __name__ == '__main__':
    main()
