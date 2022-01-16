import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise


def recognize_gesture(img,thresholded,hand_region):
    maximum_contour=cv2.convexHull(hand_region)


    up_point=tuple(maximum_contour[maximum_contour[:, :, 1].argmin()][0])
    left_point = tuple(maximum_contour[maximum_contour[:, :, 0].argmin()][0])
    down_point = tuple(maximum_contour[maximum_contour[:, :, 1].argmax()][0])
    right_point = tuple(maximum_contour[maximum_contour[:, :, 0].argmax()][0])

    # get center
    center_x = int((left_point[0]+right_point[0])/2)
    center_y = int((up_point[1] +down_point[1])/2)

    distances=pairwise.euclidean_distances([(center_x, center_y)], Y=[left_point, right_point, up_point, down_point])[0]
    distance_max=distances[distances.argmax()]


    circum= (2 * np.pi * int(0.75 * distance_max))

    circle_area = np.zeros(thresholded.shape[:2], dtype="uint8")

    cv2.circle(circle_area, (center_x, center_y), int(0.75 * distance_max), 255, 1)

    circle_area = cv2.bitwise_and(thresholded, thresholded, mask=circle_area)


    (contours, hierarchy) = cv2.findContours(circle_area.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    number = 0

    for i, c in enumerate(contours):

        #  bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)


        if ((center_x + (center_y * 0.25)) > (y + h)) and ((circum * 0.25) > c.shape[0]):
            number += 1

    return number



def recognize_hand_regin(img,grayimg,thresh=75):
    # binary image,
    t=cv2.threshold(grayimg,thresh,255,cv2.THRESH_BINARY)
    contours,hierarchy =cv2.findContours(t[1].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours)!=0:
        hand_region=max(contours,key=cv2.contourArea)
        return (t[1],hand_region)
    else:
        return None


if __name__ == "__main__":
    # get the image
    image = cv2.imread("resources/t2.jpg")

    image_copy=imutils.resize(image, width=700).copy()
    image = imutils.resize(image, width=700)
    #preprocess
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gausianblur
    gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
    # hand region

    hand_region=recognize_hand_regin(image_copy,gray_img)

    (m,n)=image.shape[:2]

    if hand_region!=None:
        (thresholded, hand_segment) = hand_region



        num_fingers = recognize_gesture(image_copy, thresholded, hand_segment)

        cv2.putText(image_copy, "This is " + str(num_fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Image", image_copy)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
