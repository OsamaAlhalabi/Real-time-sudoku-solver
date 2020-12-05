import cv2 as cv
from preprocessing import recognize_sudoku, filter_and_repair, retrieve_cells

if __name__ == "__main__":
    path = r"C:\Users\Nitro\Desktop\imgs\sudoku.jpg"
    img = cv.imread(path)
    img = recognize_sudoku(img)
    filtered_img = filter_and_repair(img)
    cells = retrieve_cells(img, filtered_img)
    # cv.imshow('output', img)
    # cv.imshow('output1', fimg)
    # cv.waitKey(0)
