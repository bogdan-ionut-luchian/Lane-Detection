import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_path = r"C:\Users\Lenovo\PycharmProjects\pythonProject\test_images\Poza5.jpg"
image1 = cv2.imread(image_path)
plt.imshow(image1)

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)
def canny(image):
    edges = cv2.Canny(image,50,150)
    return edges

def region(image):
    height, width = image.shape
    triangle = np.array([[(100, height), (475, 325), (width, height)]])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def average(image, lines):
    left = []
    right = []
    for line in lines:
        print(line)
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])

def make_points(image, average):
    slope, y_int = average
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])


def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return lines_image



plt.imshow(image1)
copy = np.copy(image1)
gray = gray(copy)
gaus = gauss(gray)
edges = canny(gaus)
isolated = region(edges)
lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average(copy, lines)
black_lines = display_lines(copy, averaged_lines)
lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
cv2.imshow("gray", gray)
cv2.imshow("gauss", gaus)
cv2.imshow("canny", edges)
cv2.imshow("iso", isolated)
cv2.imshow("lanes", lanes)
cv2.waitKey(0)



#Video
video = r"C:\Users\Lenovo\PycharmProjects\pythonProject\test_videos\1.mp4"
cap = cv2.VideoCapture(video)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gaus = gauss(frame)
        edges = cv2.Canny(gaus,50,150)
        isolated = region(edges)
        lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average(frame, lines)
        black_lines = display_lines(frame, averaged_lines)
        lanes = cv2.ad1dWeighted(frame, 0.8, black_lines, 1, 1)
        cv2.imshow("frame", lanes)
cap.release()
cv2.destroyAllWindows()