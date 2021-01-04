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

        prev_frame_time = 0
        new_frame_time = 0
        solved_grid = np.zeros((w, h))
        while ret:
            img, fuc = preprocessing.recognize_sudoku(frame)
            write_solution = np.copy(img)
            try:
                filtered, grid_img = preprocessing.filter_and_repair(img)

                sudoku, inputs = preprocessing.retrieve_cells(filtered, grid_img)

                sudoku_grid = np.zeros((9, 9), dtype=np.uint8)

                inputs, mask = preprocessing.filter_empty(inputs.reshape(81, 64, 64))

                outputs = features.predict(inputs, False)
                try:
                    logging.info(f'accuracy: {accuracy_score(ground_truth, outputs)}')
                except ValueError as e:
                    logging.error(e)

                sudoku_grid[mask] = outputs

                solved = solve(sudoku_grid)

                write_solution = preprocessing.write_solution_on_image(write_solution, solved, mask)

                solved_grid = cv.warpPerspective(write_solution, fuc,
                                                 (frame.shape[1], frame.shape[0])
                                                 , flags=cv.WARP_INVERSE_MAP)

            except preprocessing.GridError as e:
                logging.error(e)

            new_frame_time = time.time()
            fps = 1 // (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            logging.info(f'fps: {fps}')

            # solved_grid = np.where(solved_grid.sum(axis=-1, keepdims=True) != 0, solved_grid, frame)
            cv.imshow("processed image", solved_grid)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()
    finally:
        cap.release()
        cv.destroyAllWindows()
        features.executor.shutdown()
        preprocessing.executor.shutdown()
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
    # inputs, mask = preprocessing.filter_empty(inputs.reshape(81, 64, 64))
    #
    # # outputs = features.predict(inputs, blur=False)
    # #
    # # print(outputs)
    #
    # for i, img in enumerate(inputs):
    #     cv.imwrite(f'cell_{i}.png', img)
    #
    # cv.destroyAllWindows()
    # features.executor.shutdown()
    # preprocessing.executor.shutdown()
