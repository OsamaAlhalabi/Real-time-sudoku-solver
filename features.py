import cv2 as cv
import numpy as np
# import pickle
import concurrent.futures as futures

# from tensorflow.keras.models import load_model

sift = cv.SIFT_create()

fast = cv.FastFeatureDetector_create()

brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

orb = cv.ORB_create()

win_size = (64, 64)
block_size = (8, 8)
block_stride = (4, 4)
cell_size = (4, 4)
bins = 9

hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)

# Define parameters for our Flann Matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
search_params = dict(checks=100)

# Create the Flann Matcher object
flann = cv.FlannBasedMatcher(index_params, search_params)

templates = np.array([cv.imread(f'templates/template_{i}.png', 0) for i in range(1, 10)])

# fast_cluster_centers = pickle.load(open('fast_cluster_centers.pkl', 'rb'))
# k_fast = 5
#
# sift_cluster_centers = pickle.load(open('sift_cluster_centers.pkl', 'rb'))
# k_sift = 9

# clf = load_model('digits_clf.h5')
executor = futures.ThreadPoolExecutor(max_workers=4)


def extract_hog_features(inputs, blur=True):
    if blur:
        inputs = [cv.GaussianBlur(img, (3, 3,), None) for img in inputs]

    hog_features = np.array([hog.compute(img, None, None, None) for img in inputs])

    return hog_features


def extract_sift_feature(inputs, blur=True):
    if blur:
        inputs = [cv.GaussianBlur(img, (3, 3,), None) for img in inputs]

    return np.array([sift.detectAndCompute(img, None) for img in inputs])


def extract_fast_feature(inputs, blur=True):
    if blur:
        inputs = [cv.GaussianBlur(img, (3, 3,), None) for img in inputs]

    key_points = [fast.detect(img, None) for img in inputs]

    return np.array([sift.compute(img, kp) for img, kp in zip(inputs, key_points)])


def extract_orb_feature(inputs, blur=True):
    if blur:
        inputs = [cv.GaussianBlur(img, (3, 3,), None) for img in inputs]

    key_points = [orb.detect(img, None) for img in inputs]

    return np.array([orb.compute(img, kp) for img, kp in zip(inputs, key_points)])


sift_templates = extract_sift_feature(templates, blur=False)

fast_templates = extract_fast_feature(templates, blur=False)

orb_templates = extract_orb_feature(templates, blur=False)


def match_templates(features, matching_templates):

    def match(digits):
        ans = np.zeros((digits.shape[0]), dtype=np.uint8)
        for idx, (_, descriptor) in enumerate(digits):
            mx = 0
            if descriptor is None:
                continue
            for integer, (kp, des) in enumerate(matching_templates):
                # Obtain matches using K-Nearest Neighbor Method
                # the result 'matches' is the number of similar matches found in both images
                matches = flann.knnMatch(des.astype(np.float32), descriptor.astype(np.float32), k=2)

                # Store good matches using Lowe's ratio test
                good_matches = 0
                for u, v in matches:
                    if u.distance < 0.7 * v.distance:
                        good_matches += 1

                if mx < good_matches:
                    mx = good_matches
                    ans[idx] = integer + 1
        return ans

    n = features.shape[0]

    future1 = executor.submit(match, features[: n // 3])
    future2 = executor.submit(match, features[n // 3: 2 * n // 3])
    future3 = executor.submit(match, features[2 * n // 3:])

    return np.hstack([future1.result(), future2.result(), future3.result()])


def predict(inputs, blur=True):
    n = inputs.shape[0]
    future1 = executor.submit(extract_fast_feature, inputs[: n // 3], blur)
    future2 = executor.submit(extract_fast_feature, inputs[n // 3: 2 * n // 3], blur)
    future3 = executor.submit(extract_fast_feature, inputs[2 * n // 3:], blur)

    future4 = executor.submit(match_templates, future1.result(), fast_templates)
    future5 = executor.submit(match_templates, future2.result(), fast_templates)
    future6 = executor.submit(match_templates, future3.result(), fast_templates)

    return np.hstack([future4.result(), future5.result(), future6.result()])


# def extract_encoded_fast_feature(inputs, blur=True):
#     fast_encoded_features = np.empty((inputs.shape[0], k_fast))
#
#     if blur:
#         inputs = [cv.GaussianBlur(img, (3, 3,), None) for img in inputs]
#
#     for idx, img in enumerate(inputs):
#         h = np.zeros((k_fast,))
#
#         kp = fast.detect(img, None)
#
#         kp, des = brief.compute(img, kp)
#
#         dist2 = np.dot(fast_cluster_centers, des.transpose())
#
#         where = np.argmin(dist2, axis=0)
#
#         for i in where:
#             h[i] += 1
#
#         fast_encoded_features[idx] = h
#
#     return fast_encoded_features
#
#
# def extract_encoded_sift_feature(inputs, blur=True):
#     sift_encoded_features = np.empty((inputs.shape[0], k_sift))
#
#     if blur:
#         inputs = [cv.GaussianBlur(img, (3, 3,), None) for img in inputs]
#
#     for idx, img in enumerate(inputs):
#         h = np.zeros((k_sift,))
#
#         kp = sift.detect(img, None)
#
#         kp, des = sift.compute(img, kp)
#
#         dist2 = np.dot(sift_cluster_centers, des.transpose())
#
#         where = np.argmin(dist2, axis=0)
#
#         for i in where:
#             h[i] += 1
#
#         sift_encoded_features[idx] = h
#
#     return sift_encoded_features


# def predict(inputs_64):
#     hog_features = extract_hog_features(inputs_64)
#
#     inputs_128 = np.array([cv.resize(img, (128, 128), cv.INTER_CUBIC) for img in inputs_64])
#
#     sift_features = extract_encoded_sift_feature(inputs_128, blur=False)
#
#     fast_features = extract_encoded_fast_feature(inputs_128, blur=False)
#
#     features = np.hstack([fast_features, sift_features, hog_features])
#
#     return clf.predict(features).argmax(axis=1) + 1
