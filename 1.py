import cv2

im = cv2.imread("./benchmarkA.png", 0)
cv2.imshow("a",im)
cv2.waitKey()