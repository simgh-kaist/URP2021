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
def w2d(img, mode):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    img=cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow("asdfasfdsf",img)
    coeffs2 = pywt.dwt2(v, mode)
    #LL, (LH, HL, HH) = coeffs2
    v=pywt.idwt2(coeffs2, mode)
    v = np.float32(v)
    h = np.float32(h)
    s = np.float32(s)
    img = cv2.merge((h,s,v))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow("asdf",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

original = cv2.imread("/Users/simgh/Downloads/Code/landscape_100_stacked_result.png",-1).astype(np.float32)
w2d(original/65535, 'haar')

"""    
original = cv2.imread("/Users/simgh/Downloads/Code/landscape_100_stacked_result.png")

b,g,r = cv2.split(original)

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(b, 'haar')
LL, (LH, HL, HH) = coeffs2
#fig = plt.figure(figsize=(12, 3))
b=pywt.idwt2(coeffs2, 'haar')
#plt.imshow(resized)
cv2.imshow("asdf",b)
cv2.waitKey()
#plt.imshow(LH+HL+HH+LL)
#plt.imshow(pywt.idwt2(HH, 'haar')/65535)

for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
#plt.show()
"""