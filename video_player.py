import cv2

cap = cv2.VideoCapture('tracked.mp4')
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(33) > 0: break

cap.release()
cv2.destroyAllWindows()