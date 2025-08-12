
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open camera")
    exit()


feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)


lk_params = dict(winSize=(15,15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
if not ret:
    print("Error: can't read first frame")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # نقاط البداية


mask_draw = np.zeros_like(old_frame)

print("تشغيل Optical Flow — اضغط 'r' لإعادة تهيئة النقاط، 'q' للخروج")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is None or len(p0) < 10:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        mask_draw = np.zeros_like(frame)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None and st is not None:
        good_new = p1[st.flatten() == 1]
        good_old = p0[st.flatten() == 1]

        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            mask_draw = cv2.line(mask_draw, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)

        img = cv2.add(frame, mask_draw)

        try:
            pts = good_new.reshape(-1, 2).astype(np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        except:
            pass

        cv2.imshow('Optical Flow (small)', img)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        mask_draw = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
