import time

import cv2 as cv
import numpy as np
import preprocessing
import features
import logging

from sklearn.metrics import accuracy_score
from pySudoku import solve

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    w, h = 1280, 1024
    cap = cv.VideoCapture(1)
    try:
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
            ret, frame = cap.read()
        else:
            ret = False
            frame = np.zeros((w, h))
        ground_truth = np.array([6, 2, 5, 9, 7, 1, 2, 3, 5, 7, 2, 1, 4, 8, 9, 3, 9, 7, 6, 5, 1, 8, 1])

        # total_matching_features = np.array([(np.concatenate([fast_kp, sift_kp]), np.concatenate([fast_des, sift_des]))
        #                                     for (fast_kp, fast_des), (sift_kp, sift_des) in zip(features.fast_templates,
        #                                                                                         features.sift_templates)
        #                                     ]
        #                                    )

        prev_frame_time = 0
        new_frame_time = 0
        while ret:
            img, rec = preprocessing.recognize_sudoku(frame)
            if rec is True:
                try:
                    filtered, grid_img = preprocessing.filter_and_repair(img)

                    sudoku, inputs = preprocessing.retrieve_cells(filtered, grid_img)

                    sudoku_grid = np.zeros((9, 9), dtype=np.uint8)

                    inputs, mask = preprocessing.filter_empty(inputs.reshape(81, 128, 128))

                    # orb_features = features.extract_orb_feature(inputs)
                    #
                    # outputs = features.match_templates(orb_features, features.orb_templates)

                    # sift_features = features.extract_sift_feature(inputs, blur=False)

                    # total_features = np.array(
                    #     [(np.concatenate([fast_kp, sift_kp]), np.concatenate([fast_des, sift_des]))
                    #      for (fast_kp, fast_des), (sift_kp, sift_des) in zip(fast_features,
                    #                                                          sift_features)])

                    # outputs = features.match_templates(total_features, total_matching_features)

                    outputs = features.predict(inputs, False)

                    # where = np.where(~mask)
                    #
                    # sudoku[where] = outputs
                    #
                    # solve(sudoku)
                    #
                    # print(solve(sudoku))
                    logging.info(f'accuracy: {accuracy_score(ground_truth, outputs)}')

                except preprocessing.GridError as e:
                    logging.error(e)

            new_frame_time = time.time()
            fps = 1 // (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            logging.info(f'fps: {fps}')

            cv.imshow("processed image", img)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()
    finally:
        cap.release()
        cv.destroyAllWindows()
        features.executor.shutdown()
    # # path = r"C:\Users\Nitro\Desktop\imgs\sudoku.jpg"
    # # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image1087.jpg"
    # # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image1070.jpg"
    # # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image1062.jpg"
    # # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image1086.jpg"
    # # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image1011.jpg"
    # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image1008.jpg"
    # # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image206.jpg"
    # # path = r"C:\Users\Nitro\Desktop\1_zHZx0IJiNrLYYqW5lyck_A.png"
    # img = cv.imread(path)
    #
    # img, _ = preprocessing.recognize_sudoku(img)
    #
    # img, gird_img = preprocessing.filter_and_repair(img)
    #
    # sudoku, inputs = preprocessing.retrieve_cells(img, gird_img)
    #
    # inputs, mask = preprocessing.filter_empty(inputs)
    #
    # outputs = features.predict(inputs, blur=False)
    #
    # print(outputs)
    #
    # cv.destroyAllWindows()
    # features.executor.shutdown()
    # preprocessing.executor.shutdown()
