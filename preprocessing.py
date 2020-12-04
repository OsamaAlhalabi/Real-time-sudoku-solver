import cv2 as cv
import numpy as np
import calculation


def detect_corners(contours, max_iter=200, coefficient=1):
    while max_iter > 0 and coefficient >= 0:
        max_iter = max_iter - 1
        epsilon = coefficient * cv.arcLength(contours, True)
        poly_approx = cv.approxPolyDP(contours, epsilon, True)
        hull = cv.convexHull(poly_approx)
        if len(hull) == 4:
            return hull
        else:
            if len(hull) > 4:
                coefficient += .01
            else:
                coefficient -= .01
    return None


def draw_borders(img, contour, corners):
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for i in range(len(contour)):
        color = (255, 0, 0)
        cv.drawContours(drawing, contour, i, color)

    for i in range(len(corners)):
        color = (0, 255, 0)
        cv.drawContours(drawing, corners, i, color)

    cv.dilate(drawing, None)
    cv.imshow('Contours', drawing)


def recognize_sudoku(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    max_area = 0
    longest_contour = None

    for c in contours:
        area = cv.contourArea(c)
        if area > max_area:
            max_area = area
            longest_contour = c

    if longest_contour is None:
        return img

    corners = detect_corners(longest_contour)
    draw_borders(img, longest_contour, corners)

    if corners is None:
        return img
    rect = calculation.detect_rect_corners(corners)

    if calculation.check_rect(rect) is None:
        return img

    mat, w, h = calculation.calc_dimensions(rect)
    perspective_transformed_matrix = cv.getPerspectiveTransform(rect, mat)
    warp = cv.warpPerspective(img, perspective_transformed_matrix, (w, h))
    cv.imshow('output', warp)
    cv.waitKey(0)
