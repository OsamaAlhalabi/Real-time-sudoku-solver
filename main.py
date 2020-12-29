import cv2 as cv
import numpy as np
import preprocessing
import features
import logging

from sklearn.metrics import accuracy_score

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

        total_matching_features = np.array([(np.concatenate([fast_kp, sift_kp]), np.concatenate([fast_des, sift_des]))
                                            for (fast_kp, fast_des), (sift_kp, sift_des) in zip(features.fast_templates,
                                                                                                features.sift_templates)
                                            ]
                                           )
        while ret:
            img, rec = preprocessing.recognize_sudoku(frame)
            if rec is True:
                try:
                    filtered, grid_img = preprocessing.filter_and_repair(img)

                    sudoku, inputs = preprocessing.retrieve_cells(filtered, grid_img)

                    inputs, mask = preprocessing.filter_empty(inputs.reshape(81, 128, 128))

                    # orb_features = features.extract_orb_feature(inputs)
                    #
                    # outputs = features.match_templates(orb_features, features.orb_templates)

                    fast_features = features.extract_fast_feature(inputs, blur=False)

                    sift_features = features.extract_sift_feature(inputs, blur=False)

                    total_features = np.array(
                        [(np.concatenate([fast_kp, sift_kp]), np.concatenate([fast_des, sift_des]))
                         for (fast_kp, fast_des), (sift_kp, sift_des) in zip(fast_features,
                                                                             sift_features)])
                    outputs = features.match_templates(total_features, total_matching_features)

                    # print(inputs.shape[0], ground_truth.shape[0])

                    print(accuracy_score(y_true=ground_truth, y_pred=outputs) * 100)

                except preprocessing.GridError as e:
                    logging.error(e)

            cv.imshow("processed image", img)

            if cv.waitKey(40) & 0xFF == ord('q'):
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
    # inputs, mask = preprocessing.filter_empty(inputs.reshape(81, 128, 128))
    #
    # fast_features = features.extract_fast_feature(255 - inputs)
    #
    # outputs = features.match_templates(fast_features, features.fast_templates)
    #
    # for img, (kp, _), integer in zip(255 - inputs, fast_features, outputs):
    #     img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    #     img = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))
    #     print(integer)
    #     cv.imshow('cell', img)
    #     cv.waitKey(0)
    #
    # cv.destroyAllWindows()
