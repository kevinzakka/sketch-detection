import cv2
import imutils
import numpy as np
from PIL import Image
import pickle
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
thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("Threshold", thresh)
cv2.waitKey(0)

# find contours
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
X_test = []
for i, c in enumerate(cnts):
    # get bounding box
    rect = cv2.boundingRect(c)
    x, y, w, h = rect
    # # draw bounding box
    cv2.rectangle(image, (x, y), (x+w, y+h),(0, 255, 0), 2)
    elem = thresh[y:y+h, x:x+w]
    
    # convert to pillow image
    pil_img = Image.fromarray(elem)
    pil_img = pil_img.convert('L')
    if i == 1:
        pil_img.show()
    pil_img = pil_img.resize((28, 28))

    x = np.asarray(pil_img, dtype='float32')
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    X_test.append(x)
    # cv2.imwrite(out_dir + str(i) + '.png', elem)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

X_test = np.array(X_test)
pickle.dump(X_test, open("./data/test/X_test_6.p", "wb"))