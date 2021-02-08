import numpy as np
import cv2

stacked_img = cv2.imread('/Users/simgh/Downloads/Code/src_pics/sign_tree/IMG_002399.JPG')
sharpening_3_weak = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpening_3_strong = np.array([[-2, -2, -2], [-2, 17, -2], [-2, -2, -2]])
filtered_img1=cv2.filter2D(stacked_img, -1, sharpening_3_weak)
filtered_img2=cv2.filter2D(stacked_img, -1, sharpening_3_strong)
cv2.imshow('filtered_result1.png', filtered_img1)
cv2.imwrite('filtered_result1.png', filtered_img1)
cv2.waitKey(0)