import cv2 as cv
import numpy as np
from preprocessing import recognize_sudoku
if __name__ == "__main__":
    path = r"C:\Users\Nitro\Desktop\imgs\sudoku.jpg"
    img = cv.imread(path)
    img = recognize_sudoku(img)
    cv.imshow('output', img)
    cv.waitKey(0)