import cv2
import imutils
import numpy as np
import argparse

out_dir = '/Users/kevin/Desktop/'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
args = vars(ap.parse_args())

# read image and resize
image = cv2.imread(args["image"])
image = imutils.resize(image, width=300)

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# invert
inverted = cv2.bitwise_not(gray)

# blur
blurred = cv2.GaussianBlur(inverted, (5, 5), 0)

# threshold
thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)[1]

# find contours
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
for i, c in enumerate(cnts):
    # get bounding box
    rect = cv2.boundingRect(c)
    x, y, w, h = rect

    # draw bounding box
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)