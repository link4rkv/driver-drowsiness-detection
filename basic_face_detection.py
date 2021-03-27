import cv2 as cv

face_cascade_file = 'cascade files/haarcascade_frontalface_alt.xml'
eyes_cascade_file = 'cascade files/haarcascade_eye_tree_eyeglasses.xml'

face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile(face_cascade_file)) or not eyes_cascade.load(cv.samples.findFile(eyes_cascade_file)):
    print('Error loading cascade file')
    exit(0)

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

    faces = face_cascade.detectMultiScale(frame_gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
            #frame = cv.rectangle(frame, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (0, 255, 0), 3)
        
    cv.imshow('Face detection', frame)
    if cv.waitKey(10) == 27:
        break


