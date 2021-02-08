import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
"""
def w2d(img, mode='haar', level=1):
    imArray = cv2.imread(img)
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_BGR2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 65535;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 65535;
    imArray_H =  np.uint16(imArray_H)
    #Display result
    cv2.imwrite('image.jpg', imArray_H)
    cv2.imshow('image',imArray_H)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imArray_H
#w2d("/Users/simgh/Downloads/Code/Man_100_stacked_result.png", level=1)
"""
def w2d(img, mode, wavelet_power,sharpen):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    coeffs2 = pywt.dwt2(v, mode)
    LL, (LH, HL, HH) = coeffs2
    coeffs2_H = list(coeffs2)
    coeffs2_H[0]*=0
    v_H=pywt.idwt2(coeffs2_H, mode)
    v_sharpen = cv2.filter2D(v_H, -1, np.array([[-1 * sharpen, -1 * sharpen, -1 * sharpen], [-1 * sharpen, 1 + 8 * sharpen, -1 * sharpen],[-1 * sharpen, -1 * sharpen, -1 * sharpen]]))
    v_wavelet = (v_sharpen*wavelet_power+v)
    v_blured=cv2.GaussianBlur(v_wavelet,(5,5), 0)
    result = cv2.merge((h,s,v_blured))
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    return result

original = cv2.imread("./Man_100_stacked_result.png",-1).astype(np.float32)
result = w2d(original/65535, 'haar', 100, 0.01)

cv2.imshow("sharpen=>wavelet", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
