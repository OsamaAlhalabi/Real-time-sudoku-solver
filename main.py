import cv2 as cv
import numpy as np
import preprocessing
import features

if __name__ == "__main__":
    w, h = 1280, 1024
    cap = cv.VideoCapture(1)
    if cap.isOpened():
        cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
        ret, frame = cap.read()
    else:
        ret = False
        frame = np.zeros((w, h))

    while ret:
        img, rec = preprocessing.recognize_sudoku(frame)
        if rec is True:
            try:
                filtered, grid_img = preprocessing.filter_and_repair(img)

                sudoku, inputs = preprocessing.retrieve_cells(filtered, grid_img)

                inputs, mask = preprocessing.filter_empty(inputs.reshape(81, 128, 128))

                fast_features = features.extract_fast_feature(inputs)

                outputs = features.match_templates(fast_features, features.fast_templates)

                print(outputs)

            except preprocessing.GridError as e:
                print(e)

        cv.imshow("processed image", img)

        if cv.waitKey(40) & 0xFF == ord('q'):
            break

        ret, frame = cap.read()

    cap.release()
    cv.destroyAllWindows()
