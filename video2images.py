import cv2
import os

def video2image(file, dir):
    cap = cv2.VideoCapture(file)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(n_frames):
        success, curr = cap.read()
        if not success:
            print(i)
            break
        if not (os.path.isdir(dir)):
            os.makedirs(os.path.join(dir))
        cv2.imwrite(f'{dir}/{i}.png', curr)

video2image('../src/src_video/lightnight/IMG_002414.MOV', '../src/src_pics/lightnight')


