import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap = cv.VideoCapture(-1)

if not cap.isOpened:
    print('Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('No captured frame!')
        break
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        
    cv.imshow('Capture - Face detection', frame)
    if cv.waitKey(10) == 27:
        break


