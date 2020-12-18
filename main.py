import cv2 as cv
import numpy as np
from preprocessing import recognize_sudoku, filter_and_repair, retrieve_cells, resize, filter_empty
import features

if __name__ == "__main__":
    path = r"C:\Users\Nitro\Desktop\imgs\sudoku.jpg"
    # path = r"C:\Users\Nitro\Desktop\Screenshot 2020-12-18 162317.jpg"
    img = cv.imread(path)

    # img = cv.resize(img, (558, 563), cv.INTER_CUBIC)

    img = recognize_sudoku(img)

    cv.imshow('img', img)

    filtered_img, thresh = filter_and_repair(img)

    sudoku, cells = retrieve_cells(img, thresh)
    #
    # inputs = np.array(resize(cells, (50, 50))).reshape((81, 50, 50))
    #
    # inputs, mask = filter_empty(inputs)
    #
    # fast_features = features.extract_fast_feature(inputs)
    #
    # outputs = features.match_templates(fast_features)
    #
    # print(outputs)

    cv.imshow('grid', thresh)

    cv.imshow('filtered', filtered_img)

    cv.waitKey(0)
