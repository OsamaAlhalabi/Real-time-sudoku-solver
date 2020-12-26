import cv2 as cv
import numpy as np

sift = cv.SIFT_create()

fast = cv.FastFeatureDetector_create()

brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

win_size = (50, 50)
block_size = (8, 8)
block_stride = (6, 6)
cell_size = (8, 8)
bins = 9

hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)

# Define parameters for our Flann Matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
search_params = dict(checks=100)

# Create the Flann Matcher object
flann = cv.FlannBasedMatcher(index_params, search_params)

templates = np.array([cv.imread(f'templates/template_{i}.png', 0) for i in range(1, 10)])

def extract_hog_features(inputs):
    hog_features = np.array([hog.compute(img, None, None, None) for img in inputs])

    return hog_features


def extract_sift_feature(inputs):
    inputs = [cv.GaussianBlur(img, (3, 3,), None) for img in inputs]

    return np.array([sift.detectAndCompute(img, None) for img in inputs])


def extract_fast_feature(inputs):
    # inputs = [cv.GaussianBlur(img, (3, 3,), None) for img in inputs]

    key_points = [fast.detect(img, None) for img in inputs]

    return np.array([brief.compute(img, kp) for img, kp in zip(inputs, key_points)])


sift_templates = extract_sift_feature(templates)

fast_templates = extract_fast_feature(templates)


def match_templates(features, matching_templates):
    ans = np.zeros((features.shape[0]), dtype=np.uint8)
    for idx, (_, descriptor) in enumerate(features):
        mx = 0
        for integer, (kp, des) in enumerate(matching_templates):
            # Obtain matches using K-Nearest Neighbor Method
            # the result 'matches' is the number of similar matches found in both images
            matches = flann.knnMatch(des.astype(np.float32), descriptor.astype(np.float32), k=2)

            # Store good matches using Lowe's ratio test
            good_matches = 0
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches += 1

            if mx < good_matches:
                mx = good_matches
                ans[idx] = integer + 1
    return ans
