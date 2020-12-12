import cv2 as cv
from preprocessing import recognize_sudoku, filter_and_repair, retrieve_cells

if __name__ == "__main__":
    path = r"C:\Users\Nitro\Desktop\imgs\sudoku.jpg"
    img = cv.imread(path)
    img = recognize_sudoku(img)

    filtered_img, thresh = filter_and_repair(img)
    sudoku, cells = retrieve_cells(img, thresh)

    cv.imshow('output', img)

    cv.imshow('grid', thresh)

    cv.imshow('filtered', filtered_img)

    cv.waitKey(0)
