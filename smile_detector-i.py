# Made by KodeHak

import cv2

# Face and smile classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Choose an image to detect smiles in
img = cv2.imread('smiling-woman.jpg')

# Must convert to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Detect faces
faces = face_detector.detectMultiScale(grayscale_img)

# Find the faces
for (x, y, w, h) in faces:
    # Draw rectangle around the faces
    cv2.rectangle(img, (x, y), (x+w, y+h), (100, 200, 50), 4)

    # Get the faces
    the_face = img[y:y + h, x:x + w]
    # Convert to grayscale
    grayscale_face = cv2.cvtColor(the_face, cv2.COLOR_RGB2GRAY)

    # Detect smiles
    smiles = smile_detector.detectMultiScale(grayscale_face, scaleFactor=1.7, minNeighbors=20)

    # Find smile in faces
    for (x_, y_, w_, h_) in smiles:
        # Draw rectangle around smiles
        cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 4)


# Display the img with faces
cv2.imshow('KodeHak Smile Detector', img)
cv2.waitKey()

# Cleanup
cv2.destroyAllWindows()

print("Code Completed")