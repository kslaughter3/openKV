import numpy as np
import cv2
import imutils
import argparse
import shapedetector

# Load an color image in grayscale
img = cv2.imread('images/poeImage 022.jpg')
print("hello world")
print(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = shapedetector.ShapeDetector()

for i in reversed(range(len(cnts))):
	c = cnts[i]
	area = cv2.contourArea(c)
	if area < 1000:
		del cnts[i]
	else:
		shape = sd.detect(c)
		if shape != "quad":
			del cnts[i]

# loop over the contours
for c in cnts:
	# compute the center of the contour
	M = cv2.moments(c)
	if M["m00"] != 0:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	else:
		cX, cY = 0, 0
		cnts.remove(c)
	area = cv2.contourArea(c)
	if area < 1000:
		cnts.remove(c)
	# draw the contour and center of the shape on the image
	cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
	cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
	cv2.putText(img, str(area), (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 

cv2.imshow('img', img)
#cv2.imshow('gray', gray)
#cv2.imshow('blurred', blurred)
#cv2.imshow('thresh', thresh)
cv2.waitKey(0)


cv2.destroyAllWindows()