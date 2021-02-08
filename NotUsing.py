import numpy as np
import cv2

def ORB_Example(src1, src2):
    img1=cv2.imread(src1,cv2.COLOR_BGR2RGB)
    img2=cv2.imread(src2,cv2.COLOR_BGR2RGB)   #target

    orb=cv2.ORB_create()

    kp1, des1=orb.detectAndCompute(img1,None)
    kp2, des2=orb.detectAndCompute(img2,None)

    bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    # matches has four indexes
    # queryIdx, trainIdx, imgIdx, distance
    matches = sorted(matches,key=lambda x:x.distance)
    # small distance means more accurate matching

    img3=cv2.drawMatches(img1,kp1,img2,kp2, matches[:20], None, matchColor=[0,0,255], flags=2)
    img3=cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
    return img3

def ORB_Stack(file_list):
    orb = cv2.ORB_create()

    stacked_image = None
    first_image = None
    first_kp = None
    first_des = None
    for f in file_list:
        image = cv2.imread(f, 1)
        imageF = image.astype(np.float32) / 255
        #print(imageF.shape, image.shape)
        # Detects keypoints and computes the descriptors
        kp, des=orb.detectAndCompute(image, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if first_image is None:
            # Save keypoints for first image
            stacked_image = imageF
            first_image = image
            first_kp = kp
            first_des = des
        else:
            matches = bf.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32([first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            #Estimate perspective transformation
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

            # RANSAC : RANdom SAmple Consensus ==> Random sampling ==> reduce noise
            h, w, _ = imageF.shape
            imageF = cv2.warpPerspective(imageF, M, (w,h))
            stacked_image += imageF

    stacked_image /= len(file_list)
    stacked_image = (stacked_image*255).astype(np.uint8)
    return stacked_image
