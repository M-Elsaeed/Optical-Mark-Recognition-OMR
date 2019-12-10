from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import math


def myFunc():

    img = cv2.imread(
        'D:/learning/cv/Project/CSE365_test_cases_project_1/test_sample8.jpg', 0)

    img_edges = cv2.Canny(img, 100, 100, apertureSize=3)

    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0,
                            100, minLineLength=100, maxLineGap=5)

    x1, y1, x2, y2 = lines[0][0]

    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


warped = myFunc()

# apply Otsu's thresholding method to binarize the warped
# piece of paper
_, thresh = cv2.threshold(warped, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

thresh = cv2.morphologyEx(
    thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))

# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions

cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

# print(len(cnts))
# loop over the contours

for c in cnts:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)


# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionCnts = contours.sort_contours(questionCnts,
                                      method="top-to-bottom")[0]
correct = 0

blank_img = np.zeros(warped.shape)
cv2.drawContours(blank_img, questionCnts, -1, 255, -1)

blank_img = np.array(blank_img, dtype='uint8')

_, _, _, centroids = cv2.connectedComponentsWithStats(
    blank_img, connectivity=8)


print(centroids)
print(len(centroids))

centroids = centroids[1: len(centroids)]

print(centroids)
print(len(centroids))

centroids = centroids[np.argsort(centroids[:, 1])]

print(centroids)

maleOrFemale = "Male" if (centroids[0][0] - 1240) < 80 else "Female"
semesterCentorid = int((centroids[1][0] - 540) / 265)

print(semesterCentorid)

semester = None
if semesterCentorid == 0:
    semester = "Fall"
elif semesterCentorid == 1:
    semester = "Spring"
elif semesterCentorid == 2:
    semester = "Summer"


program = None
programNumber = int((centroids[2][0] - 440) / 135)

print(programNumber)
if centroids[2][1] > 440 and centroids[2][1] < 470:
    if programNumber == 0:
        program = "MCTA"
    elif programNumber == 1:
        program = "ENVR"
    elif programNumber == 2:
        program = "BLDG"
    elif programNumber == 3:
        program = "CESS"
    elif programNumber == 4:
        program = "ERGY"
    elif programNumber == 5:
        program = "COMM"
    elif programNumber == 6:
        program = "MANF"
else:
    if programNumber == 0:
        program = "LAAR"
    elif programNumber == 2:
        program = "MATL"
    elif programNumber == 3:
        program = "CISE"
    elif programNumber == 4:
        program = "HAUD"

questions = [[None, None, None, None, None],
             [None, None, None, None, None, None],
             [None, None, None],
             [None, None, None],
             [None, None]]

currentXPosi = 1110
currentYPos = 965


c = 3
for i in range(len(questions)):
    for j in range(len(questions[i])):
        if(centroids[c][0] < 1110):
            print(centroids[c])
        questions[i][j] = int((centroids[c][0] - 1110) / 100) + 1
        c += 1
    pass

print(maleOrFemale)
print(semester)
print(program)
print(questions)