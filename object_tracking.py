import cv2

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# إنشاء كائن الخلفية
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# لائحة لتخزين نقاط المسار
track_points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تطبيق خوارزمية اكتشاف الحركة
    fgmask = fgbg.apply(frame)

    # تحسين الماسك (إزالة الضوضاء)
    fgmask = cv2.medianBlur(fgmask, 5)

    # العثور على الأجسام (Contours)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None
    for contour in contours:
        if cv2.contourArea(contour) > 800:  # تجاهل الأجسام الصغيرة
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # حساب مركز الجسم
            center = (x + w // 2, y + h // 2)

    # إذا فيه مركز، نضيفه للمسار
    if center:
        track_points.append(center)

    # رسم المسار
    for i in range(1, len(track_points)):
        if track_points[i - 1] is None or track_points[i] is None:
            continue
        cv2.line(frame, track_points[i - 1], track_points[i], (0, 0, 255), 2)

    # عرض النتائج
    cv2.imshow("Object Tracking", frame)
    cv2.imshow("FG Mask", fgmask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
