import numpy as np
import cv2
import tictoc
import pywt
import wavelet

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
def stackImagesECC(file_list, lowres, ratio):
    global diff
    M = np.eye(3, 3, dtype=np.float32)

    first_image = None
    stacked_image = None
    iter_num=100        #iteration nuber
    term_eps= 1e-01       #termination epsilon
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iter_num, term_eps)
    i = 0
    for file in file_list:
        if lowres==True:
            image = cv2.imread(file,1).astype(np.float32) / 255
            #image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
            image = cv2.pyrDown(image)
        else:
            image = cv2.imread(file, 1).astype(np.float32) / 255

        if first_image is None:
            # convert to gray scale floating point image
            first_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            stacked_image = image
        else:
            # Estimate perspective transform
            s, M = cv2.findTransformECC(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), first_image, M, cv2.MOTION_HOMOGRAPHY, criteria, None, 5)
            w, h, _ = image.shape
            ones=np.ones((w,h))
            blackArea=cv2.warpPerspective(ones, M, (h, w))
            # Align image to first image
            image = cv2.warpPerspective(image, M, (h, w))
            image[blackArea != 1.] = stacked_image[blackArea!=1.]/i
            #diff=abs(stacked_image/i-image)
            #image[idx] = stacked_image[idx]/i
            stacked_image += image
        i+=1
    stacked_image /= len(file_list)
    stacked_image = (stacked_image*65535).astype(np.uint16)
    return stacked_image

file_list=[]
lowres=False
ratio=0.5
dataset='Man'
howmany=100
for i in range(howmany):
    file_list.append(f'./src_pics/{dataset}/{i}.png')
tictoc.tic()
stacked_img=stackImagesECC(file_list, lowres, ratio)
tictoc.toc()
sharpening_3_weak = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
strongness=1
next=-strongness*1
sharpening_3_strong =  np.array([[next, next, next], [next, abs(next*4+next*4)+1, next], [next, next, next]])

"""
b, g, r = cv2.split(stacked_img)
b_wavelet=w2d(b, level=1)
g_wavelet=w2d(g, level=1)
r_wavelet=w2d(r, level=1)
b_wavelet=cv2.filter2D(b_wavelet, -1, sharpening_3_weak)
g_wavelet=cv2.filter2D(g_wavelet, -1, sharpening_3_weak)
r_wavelet=cv2.filter2D(r_wavelet, -1, sharpening_3_weak)
"""
v = cv2.filter2D(v, -1, sharpening_3_strong)
print(v_wavelet)
sharpened=cv2.merge((h,s,v))
hsv_sharpened = cv2.cvtColor(sharpened, cv2.COLOR_HSV2BGR)
sharpened = cv2.filter2D(stacked_img, -1, sharpening_3_strong)


cv2.imshow("hsv sharpened", hsv_sharpened)
cv2.imshow("wavelet",v_wavelet)
#cv2.imshow("stacked", stacked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
#avg_kernel=np.ones((3,3))/9
sharpening_3_weak = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpening_3_strong = np.array([[-2, -2, -2], [-2, 17, -2], [-2, -2, -2]])
#sharpening_5 = np.array([[-1, -1, -1, -1, -1],[-1, 2, 2, 2, -1],[-1, 2, 9, 2, -1],[-1, 2, 2, 2, -1],[-1, -1, -1, -1, -1]]) / 9.0
blur = cv2.GaussianBlur(img,(5,5),0)
wavelet=np.array([[6, 6, 6, 6, 6],[6, 6, 6, 6, 6],[6, 6, 50, 6, 6],[6, 6, 6, 6, 6],[6, 6, 6, 6, 6]]) / 194
#wavelet2=np.array([[1, 1, 1, 1, 1],[1, 6, 6, 6, 1],[1, 6, 50, 6, 1],[1, 6, 6, 6, 1],[1, 1, 1, 1, 1]]) /114
filtered_img1=cv2.filter2D(stacked_img, -1, sharpening_3_weak)
filtered_img2=cv2.filter2D(stacked_img, -1, sharpening_3_strong)
#filtered_img2=cv2.filter2D(stacked_img, -1, wavelet2)
#filtered_img=cv2.Laplacian(filtered_img,-1)

#cv2.imshow('stacked_result', stacked_img)
if lowres==True:
    cv2.imwrite(f'lowres_{dataset}_{howmany}_stacked_result.png', stacked_img, [cv2.CV_16U])
    cv2.imwrite(f'lowres_{dataset}_{howmany}_filtered.png', filtered_img1, [cv2.CV_16U])
else:
    cv2.imwrite(f'{dataset}_{howmany}_stacked_result.png', stacked_img, [cv2.CV_16U])
    cv2.imwrite(f'{dataset}_{howmany}_filtered.png', filtered_img1, [cv2.CV_16U])



"""