import cv2
import os

camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

detector = cv2.CascadeClassifier('frontal.xml')

person_id = input('Enter ID:')

print('Look at the camera please!')

count = 0
flag = 0

while True:
    ret, image = camera.read()
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(grayscale_image, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        cv2.imwrite('dataset/person-' + str(person_id) + '-' + str(count) + '.jpg', grayscale_image[y:y+h, x:x+w])

        cv2.imshow('image', image)

        # print(count)
        if count == 30:
            flag = 1
            break

    if flag == 1:
        break


print('Done!')
camera.release()
cv2.destroyAllWindows()