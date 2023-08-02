import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly

img = cv2.imread("C:/Users/Skylar/Desktop/27count.jpg")
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(10,10))
plt.axis("off")
plt.imshow(img1)
plt.show()

box, label, count = cv.detect_common_objects(img)
output = draw_bbox(img, box, label, count)

output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(10,10))
plt.axis("off")
plt.imshow(output)
plt.show()
print(count)