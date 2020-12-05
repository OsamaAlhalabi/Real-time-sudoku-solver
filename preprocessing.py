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


def recognize_sudoku(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 23, 5)
    img2, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    areas = np.array(list(map(cv.contourArea, contours)))

    longest_contour = contours[np.argmax(areas)]

    corners = detect_corners(longest_contour)
    draw_borders(img, longest_contour, corners)

    if corners is None:
        return img

    rect = calculation.detect_rect_corners(corners)

    if not calculation.check_rect(rect):
        return img

    mat, w, h = calculation.calc_dimensions(rect)
    perspective_transformed_matrix = cv.getPerspectiveTransform(rect, mat)
    warp = cv.warpPerspective(img, perspective_transformed_matrix, (w, h))
    return warp


def filter_and_repair(img):
    img, contours, hierarchy = cv.findContours(img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv.contourArea(c)
        if area < 1000:
            cv.drawContours(img, [c], -1, (0, 0, 0), -1)

    # Fix horizontal and vertical lines
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 5))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, vertical_kernel, iterations=7)

    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, horizontal_kernel, iterations=5)

    return img


def retrieve_cells(img, thresh):
    # Sort by top to bottom and each row by left to right
    invert = 255 - thresh
    _, contours, _ = cv.findContours(invert, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, _ = calculation.sort_contours(contours, method="top-to-bottom")

    sudoku_rows = []
    row = []
    cells = []
    for (i, c) in enumerate(contours, 1):
        area = cv.contourArea(c)
        if area < 3000:
            row.append(c)
            if i % 9 == 0:
                contours, bounding_boxes = calculation.sort_contours(row, method="left-to-right")
                sudoku_rows.append(contours)
                row.clear()
                cells.extend([img[y: y + h, x: x + w] for (x, y, w, h) in bounding_boxes])

    # Iterate through each box
    for row in sudoku_rows:
        for c in row:
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv.drawContours(mask, [c], -1, (255, 255, 255), -1)
            result = cv.bitwise_and(img, mask)
            result[mask == 0] = 255
            cv.imshow('result', result)
            cv.waitKey(175)

    return cells
