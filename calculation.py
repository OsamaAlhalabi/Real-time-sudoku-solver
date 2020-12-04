import numpy as np
import math


def calc_angle(vector_1, vector_2):
    return np.arccos(np.dot(vector_1 / np.linalg.norm(vector_1), vector_2 / np.linalg.norm(vector_2))) * 57.2958


def check_right_angle(angle, epsilon):
    return abs(angle - 90) < epsilon


def side_lengths_are_too_different(A, B, C, D, eps_scale):
    AB = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
    AD = math.sqrt((A[0] - D[0]) ** 2 + (A[1] - D[1]) ** 2)
    BC = math.sqrt((B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2)
    CD = math.sqrt((C[0] - D[0]) ** 2 + (C[1] - D[1]) ** 2)
    shortest = min(AB, AD, BC, CD)
    longest = max(AB, AD, BC, CD)
    return longest > eps_scale * shortest


def check_rect(rect):
    ab = rect[1] - rect[0]
    ad = rect[3] - rect[0]
    bc = rect[2] - rect[1]
    dc = rect[2] - rect[3]

    if not (check_right_angle(calc_angle(ab, ad), 20)) and check_right_angle(calc_angle(ab, bc),
                                                                             20) and check_right_angle(
        calc_angle(bc, dc),
        20) and check_right_angle(
        calc_angle(ad, dc), 20):
        return None
    if side_lengths_are_too_different(rect[0], rect[1], rect[2], rect[3], 1.2):
        return None
    return True


def detect_rect_corners(corners):
    rect = np.zeros((4, 2), dtype="float32")
    corners = corners.reshape(4, 2)
    index = 0
    max_val = 1000
    for i in range(4):
        if corners[i][0] + corners[i][1] < max_val:
            max_val = corners[i][0] + corners[i][1]
            index = i
    rect[0] = corners[index]
    corners = np.delete(corners, index, 0)

    min_val = 0
    for i in range(3):
        if corners[i][0] + corners[i][1] > min_val:
            min_val = corners[i][0] + corners[i][1]
            index = i
    rect[2] = corners[index]
    corners = np.delete(corners, index, 0)

    if corners[0][0] > corners[1][0]:
        rect[1] = corners[0]
        rect[3] = corners[1]

    else:
        rect[1] = corners[1]
        rect[3] = corners[0]

    return rect.reshape(4, 2)


def calc_dimensions(rect):
    (tl, tr, br, bl) = rect
    width = max(int(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))),
                int(np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))))
    height = max(int(np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))),
                 int(np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))))
    return np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32"), width, height
