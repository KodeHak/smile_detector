# Made by KodeHak

import cv2

# Face and smile classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Choose an image to detect smiles in
webcam = cv2.VideoCapture(0)

# Iterate over frames
while True:
    # Read frames
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(grayscale_frame)

    # Find the faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        # Get the faces
        the_face = frame[y:y + h, x:x + w]
        # Convert to grayscale
        grayscale_face = cv2.cvtColor(the_face, cv2.COLOR_RGB2GRAY)

        # Detect smiles
        smiles = smile_detector.detectMultiScale(grayscale_face, scaleFactor=1.7, minNeighbors=20)

        # Find smile in faces
        for (x_, y_, w_, h_) in smiles:
            # Draw rectangle around smiles
            cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 4)


    # Display frames with smiles
    cv2.imshow('KodeHak Smile Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()

print("Code Completed")