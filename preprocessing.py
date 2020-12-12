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
        elif len(hull) > 4:
            coefficient += .01
        else:
            coefficient -= .01
    return None


def draw_borders(img, contours, corners):
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    cv.drawContours(drawing, contours, -1, (255, 0, 0))

    cv.drawContours(drawing, corners, -1, (0, 255, 0))

    cv.dilate(drawing, None)


def recognize_sudoku(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 23, 5)
    _, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

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

    thresh, contours, hierarchy = cv.findContours(img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv.contourArea(c) < 1000]

    cv.drawContours(thresh, contours, -1, (0, 0, 0), -1)

    # Fix horizontal and vertical lines
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 5))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, vertical_kernel, iterations=7)

    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, horizontal_kernel, iterations=5)

    return cv.bitwise_or(thresh, img), thresh


def sort_contours(contours, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    bounding_boxes = list(map(cv.boundingRect, contours))

    contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes), key=lambda obj: obj[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return contours, bounding_boxes


def sort_bounding_boxes(bounding_boxes, method='left-to-right'):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    return sorted(bounding_boxes, key=lambda obj: obj[i], reverse=reverse)


def retrieve_cells(img, thresh):

    invert = 255 - thresh
    _, contours, _ = cv.findContours(invert, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    areas = list(map(cv.contourArea, contours))

    cell_area = invert.shape[0] * invert.shape[1] / 81 + 1e-5

    contours = [c for c, area in zip(contours, areas) if area < cell_area]

    # Sort by top to bottom and each row by left to right
    _, bounding_boxes = sort_contours(contours, method="top-to-bottom")

    sudoku = [sort_bounding_boxes(bounding_boxes[i: i + 9], method='left-to-right') for i in range(0, len(contours), 9)]

    cells = [[img[y: y + h, x: x + w] for x, y, w, h in row_boxes] for row_boxes in sudoku]

    # # Iterate through each box
    # for row in sudoku:
    #     for x, y, w, h in row:
    #         mask = np.zeros(img.shape, dtype=np.uint8)
    #         cv.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
    #         result = cv.bitwise_and(img, mask)
    #         result[mask == 0] = 255
    #         cv.imshow('result', result)
    #         cv.waitKey(175)

    return sudoku, cells

