import cv2
import numpy as np
import keras


def nothing1(x):
    pass


image_x, image_y = 64, 64
from keras.models import load_model


def predictor():
    from keras import backend as K
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img(r"G:\\Code_dataset\1.png",target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    #classifier = load_model(r'G:\\Code_dataset\Handgesture_Final_FCN.h5')
    classifier = load_model(r'G:\\Code_dataset\Handgesture_Final_CNN.h5')
    #classifier = load_model(r'G:\\Code_dataset\Handgesture_Final_VGG16.h5')

    result = classifier.predict(test_image)
    K.clear_session()
   # print(result)
    if result[0][0] == max(result[0]):
        return ' Zero  '
    elif result[0][1] == max(result[0]):
        return ' One  '
    elif result[0][2] == max(result[0]):
        return ' Two  '
    elif result[0][3] == max(result[0]):
        return ' Three  '
    elif result[0][4] == max(result[0]):
        return '  Four  '
    elif result[0][5] == max(result[0]):
        return 'Five'
    elif result[0][6] == max(result[0]):
        return 'Six'
    elif result[0][7] == max(result[0]):
        return 'Seven'
    elif result[0][8] == max(result[0]):
        return 'Eight'
    elif result[0][9] == max(result[0]):
        return 'Nine'
    elif result[0][10] == max(result[0]):
        return 'A'
    elif result[0][11] == max(result[0]):
        return 'B'
    elif result[0][12] == max(result[0]):
        return 'C'
    elif result[0][13] == max(result[0]):
        return 'D'
    elif result[0][14] == max(result[0]):
        return 'E'
    elif result[0][15] == max(result[0]):
        return 'F'
    elif result[0][16] == max(result[0]):
        return 'I'
    elif result[0][17] == max(result[0]):
        return 'J'
    elif result[0][18] == max(result[0]):
        return 'K'
    elif result[0][19] == max(result[0]):
        return 'L'
    elif result[0][20] == max(result[0]):
        return 'U'
    elif result[0][21] == max(result[0]):
        return 'M'
    elif result[0][22] == max(result[0]):
        return 'N'
    elif result[0][23] == max(result[0]):
        return 'O'
    elif result[0][24] == max(result[0]):
        return 'P'
    elif result[0][26] == max(result[0]):
        return 'Q'
    elif result[0][27] == max(result[0]):
        return 'R'
    elif result[0][28] == max(result[0]):
        return 'S'
    elif result[0][29] == max(result[0]):
        return 'T'
    elif result[0][30] == max(result[0]):
        return 'U'
    elif result[0][31] == max(result[0]):
        return 'V'
    elif result[0][32] == max(result[0]):
        return 'W'
    elif result[0][33] == max(result[0]):
        return 'X'
    elif result[0][34] == max(result[0]):
        return 'Y'
    elif result[0][35] == max(result[0]):
        return 'Z'
    


print(predictor())
import pyttsx3

hey = pyttsx3.init()

cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
ls = []
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing1)
cv2.createTrackbar("L - S", "Trackbars", 20, 255, nothing1)
cv2.createTrackbar("L - V", "Trackbars", 80, 255, nothing1)
cv2.createTrackbar("U - H", "Trackbars", 20, 179, nothing1)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing1)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing1)

cv2.namedWindow("test")

img_counter = 0
a = ''
img_text = ''
while True:
    ret, frame = cam.read()
    if not ret:
        break
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
    imcrop = img[100:300, 425:625]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    print('>>>>>>>>>>>>>',hsv.shape)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blur = cv2.GaussianBlur(mask, (3,3), 0)
    mask = cv2.medianBlur(mask, 15)
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ls.append(img_text)
    if len(ls) > 5 and ls[-1] == ls[-2] and ls[-3] == ls[-2] and ls[-3] == ls[-4] and ls[-4] == ls[-5]:
    #for len(ls) > 5 and ls[-1] == ls[-2] and ls[-3] == ls[-2] and ls[-3] == ls[-4] and ls[-4] == ls[-5]:
        print(ls[-1])
        hey.say("{}".format(ls[-1]))
        a = a + ls[-1]
        ls = []


        
    cv2.putText(frame, a, (30, 450), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("test", frame)
    cv2.imshow("mask", mask)

    #if cv2.waitKey(1) == ord('c'):

    img_name = r"G:\\Code_dataset\1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)

    img_text = predictor()
    if cv2.waitKey(1) == ord('s'):
        hey.say("{}".format(a))
        hey.runAndWait()
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
