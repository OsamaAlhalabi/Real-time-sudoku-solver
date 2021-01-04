import cv2 as cv
import numpy as np
import calculation
import concurrent.futures as futures
import math

executor = futures.ThreadPoolExecutor(max_workers=8)


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

    areas = list(map(cv.contourArea, contours))

    if not areas:
        return img, None

    longest_contour = contours[np.argmax(areas)]

    # if longest_contour is None:
    #     return img, False

    corners = detect_corners(longest_contour)
    draw_borders(img, longest_contour, corners)

    if corners is None:
        return img, False

    rect = calculation.detect_rect_corners(corners)

    # if not calculation.check_rect(rect):
    #     return img, False

    mat, w, h = calculation.calc_dimensions(rect)
    perspective_transformed_matrix = cv.getPerspectiveTransform(rect, mat)
    warp = cv.warpPerspective(img, perspective_transformed_matrix, (w, h))

    return warp, perspective_transformed_matrix


def write_solution_on_image(image, grid, mask):
    width = image.shape[1] // 9
    height = image.shape[0] // 9
    for i in range(9):
        for j in range(9):
            if mask[i][j]:
                continue
            text = str(grid[i][j])
            off_set_x = width // 15
            off_set_y = height // 15

            (text_height, text_width), base_line = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX,
                                                                  fontScale=1, thickness=3)
            font_scale = 0.6 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width * j + math.floor((width - text_width) / 2) + off_set_x
            bottom_left_corner_y = height * (i + 1) - math.floor((height - text_height) / 2) + off_set_y
            img = cv.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y),
                             cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness=3, lineType=cv.LINE_AA)
    return img


def filter_and_repair(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    thresh = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    thresh, contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    w, h = img.shape
    cell_area = (w / 9 - 16) * (h / 9 - 16) + 1e-5

    contours = [c for c in contours if cv.contourArea(c) < cell_area]

    cv.drawContours(thresh, contours, -1, (0, 0, 0), -1)

    # Fix horizontal and vertical lines
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 7))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, vertical_kernel, iterations=5)
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 1))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, horizontal_kernel, iterations=5)

    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    filtered = cv.bitwise_or(thresh, img)

    return filtered, thresh


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


class GridError(Exception):
    pass


def fix_cell(cell):
    img = cv.resize(cell, (64, 64), cv.INTER_CUBIC)
    return cv.rectangle(img, (0, 0), img.shape, (0, 0, 0), 7)


def retrieve_cells(img, thresh):
    invert = 255 - thresh
    _, contours, _ = cv.findContours(invert, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    areas = list(map(cv.contourArea, contours))

    peris = [cv.arcLength(c, True) for c in contours]

    approxs = [cv.approxPolyDP(c, 0.017 * peri, True) for c, peri in zip(contours, peris)]

    w, h = invert.shape

    img_area = w * h

    # cell_area_lower_bound = (w / 9 - 16) * (h / 9 - 16) + 1e-5
    #
    # cell_area_upper_bound = w * h / 81 + 1e-5

    contours = [c for c, area, approx in zip(contours, areas, approxs)
                if 0.0001 < area / img_area < 0.02 and len(approx) == 4]

    # contours = [c for c, area in zip(contours, areas) if cell_area_lower_bound < area < cell_area_upper_bound]

    if len(contours) != 81:
        raise GridError(f'Grid error. found {len(contours)} cell only!')

    # Sort by top to bottom and each row by left to right
    _, bounding_boxes = sort_contours(contours, method="top-to-bottom")

    sudoku = [sort_bounding_boxes(bounding_boxes[i: i + 9], method='left-to-right') for i in range(0, len(contours), 9)]

    cells = np.array([[fix_cell(img[y: y + h, x: x + w]) for x, y, w, h in row_boxes] for row_boxes in sudoku])

    def connected_component_filtering(cells_slice):
        for row in cells_slice:
            for cell in row:
                _, label_img = cv.connectedComponents(cell)
                flatten = label_img.flatten()
                uniques, counts = np.unique(flatten[flatten > 0], return_counts=True)
                if counts.size:
                    color_idx = np.argmax(counts)
                    color = uniques[color_idx]
                    mask = np.zeros_like(label_img, dtype=np.uint8)
                    mask[label_img == color] = 255
                    cv.bitwise_and(cell, mask, cell)

        return cells_slice

    splits = np.hsplit(cells, 3)
    splits = np.vsplit(np.concatenate(splits), 9)

    future_objects = [executor.submit(connected_component_filtering, s) for s in splits]

    cells = np.vstack([future.result() for future in future_objects])

    cells = np.hstack(np.split(cells, 3))

    # # Iterate through each box
    # for row in sudoku:
    #     for x, y, w, h in row:
    #         mask = np.full(img.shape, 255, dtype=np.uint8)
    #         cv.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 0), -1)
    #         cv.imshow('result', img | mask)
    #         cv.waitKey(175)
    # cv.destroyWindow('result')

    return sudoku, cells


def filter_empty(inputs):
    slices = np.array_split(inputs, 9)

    def filtering(inputs_slice):
        mask = np.array([(cv.countNonZero(img) / (img.shape[0] * img.shape[1])) > 0.05 for img in inputs_slice])
        return inputs_slice[mask], mask

    future_objects = [executor.submit(filtering, s) for s in slices]
    ret = np.vstack([future.result()[0] for future in future_objects])
    ret_mask = np.vstack([future.result()[1] for future in future_objects])
    return ret, ret_mask
