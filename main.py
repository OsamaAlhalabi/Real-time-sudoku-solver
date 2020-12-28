import cv2 as cv
import preprocessing
import features

if __name__ == "__main__":
    # path = r"C:\Users\Nitro\Desktop\imgs\sudoku.jpg"
    # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image1087.jpg"
    # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image1070.jpg"
    # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image1062.jpg"
    # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image1086.jpg"
    # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image1011.jpg"
    # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image1008.jpg"
    # path = r"C:\Users\Nitro\Downloads\Compressed\v2_train\image206.jpg"
    # path = r"C:\Users\Nitro\Desktop\1_zHZx0IJiNrLYYqW5lyck_A.png"

    cap = cv.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    while ret:
        fet, frame = cap.read()

        img, rec = preprocessing.recognize_sudoku(frame)
        if rec is True:
            try:
                filtered, gird_img = preprocessing.filter_and_repair(img)

                sudoku, inputs = preprocessing.retrieve_cells(filtered, gird_img)

                inputs, mask = preprocessing.filter_empty(inputs.reshape(81, 128, 128))

                fast_features = features.extract_fast_feature(inputs)

                outputs = features.match_templates(fast_features, features.fast_templates)
            except preprocessing.GridError as e:
                print(e)
        cv.imshow("inter", img)
        if cv.waitKey(40) == 27:
            break

    # for img,  (kp, _), integer in zip(inputs, fast_features, outputs):
    #     img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    #     img = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))
    #     print(integer)
    #     cv.imshow('cell', img)
    #     cv.waitKey(0)
    #
    cv.destroyAllWindows()
