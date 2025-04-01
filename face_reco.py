import cv2

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open webcam
webcam = cv2.VideoCapture(0)  

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, img = webcam.read()
    
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('Face Detection', img)

    key = cv2.waitKey(10)
    if key == 27:  # Escape key to exit
        break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
