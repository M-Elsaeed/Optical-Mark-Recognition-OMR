import numpy as np
import cv2
import math
import sys
import os

# Taking the path of the answer sheet
imagePath = input(
    "Please input the RELATIVE path of the image you want to analyze without quotes\n")
fullImagePath = os.getcwd().replace('\\', '/') + '/' + imagePath
print(fullImagePath)


# Returns the read image with correct orientation in greyscale
def retrieveImageWithOrientation():

    img = cv2.imread(fullImagePath, 0)
    # 'D:/learning/cv/Optical-Mark-Recognition-OMR-/CSE365_test_cases_project_1/test_sample1.jpg'

    # Obtain edges
    img_edges = cv2.Canny(img, 100, 100, apertureSize=3)

    # Obtining lines in the image
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0,
                            100, minLineLength=100, maxLineGap=5)

    # Through testing of a single line, we can determine the angle of rotation of the image
    x1, y1, x2, y2 = lines[0][0]
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

    # Applying affine rotation transformation with the pre-calculated angle to get an upright image
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


# Retrieving the image with correct upright orientation
orientedImage = retrieveImageWithOrientation()

# Getting Binary image
_, thresh = cv2.threshold(orientedImage, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Opening Image to remove small pixels/noise
thresh = cv2.morphologyEx(
    thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))

# Obtaining centroids through connected components
_, _, _, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

# Removing background connected component
centroids = centroids[1: len(centroids)]

# Sorting components by y-coordinate
centroids = centroids[np.argsort(centroids[:, 1])]

# Next few lines create three arrays for centroids lying in their respective areas
# We do this to ignore blobs outside our area of intereset,
# another benefit is to detect duplicate answers for the same question


genderCentroids = []
for i in range(len(centroids)):
    if centroids[i][0] >= 1200 and centroids[i][0] <= 1570 and centroids[i][1] >= 260 and centroids[i][1] <= 320:
        genderCentroids.append(centroids[i])

semesterCentroids = []
for i in range(len(centroids)):
    if centroids[i][0] >= 340 and centroids[i][0] <= 1570 and centroids[i][1] >= 345 and centroids[i][1] <= 405:
        semesterCentroids.append(centroids[i])

programCentroids = []
for i in range(len(centroids)):
    if centroids[i][0] >= 340 and centroids[i][0] <= 1570 and centroids[i][1] >= 430 and centroids[i][1] <= 525:
        programCentroids.append(centroids[i])

questionsCentroids = []
for i in range(len(centroids)):
    if centroids[i][0] >= 1100 and centroids[i][0] <= 1560 and centroids[i][1] >= 930 and centroids[i][1] <= 2075:
        questionsCentroids.append(centroids[i])


# Obtaining Gender through location tests
gender = None
if(len(genderCentroids) == 1):
    gender = "Male" if (genderCentroids[0][0] - 1240) < 80 else "Female"

# Obtaining semester through location tests
semester = None
if(len(semesterCentroids) == 1):
    semesterCentorid = int((semesterCentroids[0][0] - 540) / 265)
    if semesterCentorid == 0:
        semester = "Fall"
    elif semesterCentorid == 1:
        semester = "Spring"
    elif semesterCentorid == 2:
        semester = "Summer"

# Obtaining program through location tests
program = None
if(len(programCentroids) == 1):
    programNumber = int((programCentroids[0][0] - 440) / 135)
    if programCentroids[0][1] > 440 and programCentroids[0][1] < 470:
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
        elif programNumber == 1:
            program = "MATL"
        elif programNumber == 2:
            program = "CISE"
        elif programNumber == 3:
            program = "HAUD"


# Data structure to hold ansewers to questions, initialized to None. Remains None if there is no answer

questions = [[None, None, None, None, None],
             [None, None, None, None, None, None],
             [None, None, None],
             [None, None, None],
             [None, None]]

# Checking for answers in the answers section

c = 0
for i in range(len(questions)):
    for j in range(len(questions[i])):
        questions[i][j] = int((questionsCentroids[c][0] - 1110) / 100) + 1
        c += 1
    pass


outputString = f'Image   : {fullImagePath}\nGender  : {gender}\nSemester: {semester}\nProgram : {program}\nAnswers :'
for i in range(len(questions)):
    outputString += f'\n\tSection {i+1}'
    for j in range(len(questions[i])):
        outputString += f'\n\t\tQ{j+1}. {questions[i][j]}'
    outputString += "\n"


print(outputString)
f = open("Analysis Output.txt", "wt")
f.write(outputString)
f.close()

input("Press Enter to exit.")
