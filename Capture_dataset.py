import cv2
import numpy as np

def nothing():
    pass

image_x, image_y = 64, 64

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

# Counters
img_counter = 0
t_counter = 1
training_set_image_name = 1
validation_set_image_name = 1
test_set_image_name = 1
NumClasses = list(range(36)) #Number of classes = 36

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 20, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 80, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 20, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

for loop in NumClasses:
    while True:

        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

        lower_blue = np.array([l_h, l_s, l_v])
        upper_blue = np.array([u_h, u_s, u_v])
        imcrop = img[102:298, 427:623]
        hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        #result = cv2.bitwise_and(imcrop, imcrop, mask=mask)

        cv2.putText(frame, "Image="+str(img_counter) +", Class "+str(loop+1), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("test", frame)
        cv2.imshow("mask", mask)

# FH : Converting to Grayscale
#        gray = cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY)

# To collect picture press 'c'
        if cv2.waitKey(1) == ord('c'):

            # if t_counter <= 150:
            if t_counter <= 4:
                img_name = "C:/Users/fahim/Desktop/ENEL 645/Final Project/Dataset/TrainImages/{}/{}.png".format(loop+1,training_set_image_name)
                save_img = cv2.resize(mask, (image_x, image_y))
                cv2.imwrite(img_name, save_img)
                print("Training image {} written!".format(training_set_image_name))
                training_set_image_name += 1

            if t_counter > 150 and t_counter <= 200:
#            if t_counter > 4 and t_counter <= 7:
                img_name = "C:/Users/fahim/Desktop/ENEL 645/Final Project/Dataset/ValidationImages/{}/{}.png".format(loop+1,validation_set_image_name)
                save_img = cv2.resize(mask, (image_x, image_y))
#                save_img = cv2.resize(gray, (image_x, image_y))
                cv2.imwrite(img_name, save_img)
                print("Validation image {} written!".format(validation_set_image_name))
                validation_set_image_name += 1
                
            if t_counter > 200 and t_counter <= 250:
#            if t_counter > 7 and t_counter <= 10:
                img_name = "C:/Users/fahim/Desktop/ENEL 645/Final Project/Dataset/TestImages/{}/{}.png".format(loop+1,test_set_image_name)
                save_img = cv2.resize(mask, (image_x, image_y))
                cv2.imwrite(img_name, save_img)
                print("test image {} written!".format(test_set_image_name))
                test_set_image_name += 1
                
            if t_counter > 200:
#            if t_counter > 10:
                t_counter = 1 # Resetting the counter
                break

            t_counter += 1
            img_counter += 1

        # To quit loop during data collection press 'q'
        elif cv2.waitKey(1) == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()