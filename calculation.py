import numpy as np


def calc_angle(u: np.ndarray, v: np.ndarray):
    return np.arccos(np.dot(u / np.linalg.norm(u), v / np.linalg.norm(v))) * 57.2958


def check_right_angle(angle: float, epsilon: float):
    return abs(angle - 90) < epsilon


def side_lengths_are_too_different(u: np.array, v: np.array, t: np.array, n: np.array, eps_scale: float):
    uv = np.dot(u - v, u - v)
    un = np.dot(u - n, u - n)
    vt = np.dot(v - t, v - t)
    tn = np.dot(t - n, t - n)

    shortest = min(uv, un, vt, tn)
    longest = max(uv, un, vt, tn)
    return longest > eps_scale * shortest


def check_rect(rect: np.ndarray):
    ab = rect[1] - rect[0]
    ad = rect[3] - rect[0]
    bc = rect[2] - rect[1]
    dc = rect[2] - rect[3]

    if not (check_right_angle(calc_angle(ab, ad), 20)):
        if check_right_angle(calc_angle(ab, bc), 20):
            if check_right_angle(calc_angle(bc, dc), 20):
                if check_right_angle(calc_angle(ad, dc), 20):
                    return False
    # check if it is a square grid ..
    if side_lengths_are_too_different(rect[0], rect[1], rect[2], rect[3], 1.44):
        return False

    return True


def detect_rect_corners(corners: np.array):
    rect = np.zeros((4, 2), dtype="float32")
    corners = corners.reshape(4, 2)

    corners_sum = np.sum(corners, axis=1)

    min_val_idx = np.argmin(corners_sum)

    rect[0] = corners[min_val_idx]
    corners = np.delete(corners, min_val_idx, 0)

    max_val_idx = np.argmax(corners_sum)

    if max_val_idx <= 1:
        rect[2] = corners[max_val_idx]
        corners = np.delete(corners, max_val_idx, 0)

    if corners[0][0] > corners[1][0]:
        rect[1] = corners[0]
        rect[3] = corners[1]
    else:
        rect[1] = corners[1]
        rect[3] = corners[0]

    return rect.reshape(4, 2)


def calc_dimensions(rect):
    u, v, t, n = rect
    width = max(int(np.linalg.norm(t - n)), int(np.linalg.norm(u - v)))

    height = max(int(np.linalg.norm(v - t)), int(np.linalg.norm(u - n)))

    return np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32"), width, height
