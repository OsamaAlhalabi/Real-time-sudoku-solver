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
    cap = cv.VideoCapture(0)
    try:
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
            _, frame0 = cap.read()
            _, frame1 = cap.read()
            ret, frame2 = cap.read()
        else:
            ret = False
            frame1 = np.zeros((w, h))
            frame2 = np.zeros_like(frame1)

        prev_frame_time = 0
        new_frame_time = 0
        while ret:
            d = cv.absdiff(frame1, frame2)
            grey = cv.cvtColor(d, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(grey, (5, 5), None)
            ret, th = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
            dilated = cv.dilate(th, np.ones((7, 7), dtype=np.uint8), iterations=9)
            img, contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            for c in contours:
                x, y, w, h = cv.boundingRect(c)
                frame = frame1[y: y + h, x: x + w]
                img, t = preprocessing.recognize_sudoku(frame)
                try:
                    filtered, grid_img = preprocessing.filter_and_repair(img)

                    sudoku, inputs = preprocessing.retrieve_cells(filtered, grid_img)

                    sudoku_grid = np.zeros((9, 9), dtype=np.uint8)

                    inputs, mask = preprocessing.filter_empty(inputs.reshape(81, 64, 64))

                    outputs = features.predict(inputs, False)

                    sudoku_grid[mask] = outputs

                    preprocessing.write_solution_on_image(sudoku, grid_img, solve(sudoku_grid), mask)

                    img = cv.warpPerspective(grid_img, t,
                                             (frame.shape[1], frame.shape[0])
                                             , flags=cv.WARP_INVERSE_MAP)
                    invert = 255 - img

                    where = np.where(invert != 0)

                    img = cv.cvtColor(invert, cv.COLOR_GRAY2BGR)

                    img[where] = frame[where]

                    frame1[y: y + h, x: x + w] = img

                    cv.imshow('solution', filtered | grid_img)

                except preprocessing.GridError as e:
                    logging.error(e)

            new_frame_time = time.time()
            fps = 1 // (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            logging.info(f'fps: {fps}')

            cv.imshow("camera stream", frame1)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            frame1 = frame2
            ret, frame2 = cap.read()
    finally:
        cap.release()
        cv.destroyAllWindows()
        features.executor.shutdown()
        preprocessing.executor.shutdown()
