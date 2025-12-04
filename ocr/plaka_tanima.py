import cv2
import numpy as np
import pytesseract
import imutils


pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


img= cv2.imread("/home/huawei/Documents/internship/Tasks/ocr/Plaka Tanıma/images/car2.jpg")
img=cv2.resize(img,(600,400))

gri = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
filtre = cv2.bilateralFilter(gri,7,200,200)
kose = cv2.Canny(filtre,40,200)# Kenar tespiti


kontur,a = cv2.findContours(kose, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = imutils.grab_contours((kontur,a))
cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[:10]

ekran = 0

for i in cnt:
    eps = 0.018 * cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, eps, True)
    if len(approx) == 4:
        ekran = approx
        break
maske = np.zeros(gri.shape, np.uint8)
yeni_maske = cv2.drawContours(maske, [ekran], 0,(255,255,255),-1)

yazi = cv2.bitwise_and(img, img, mask=maske)

(x,y) = np.where(maske==255)# MAskedeki beyaz piksellerin koordinatlarını alır
(ustx,usty) = (np.min(x), np.min(y))
(altx,alty) = (np.max(x), np.max(y))

kirp = gri[ustx:altx+1, usty:alty+1] # +1 ekledik çünkü max değeri de dahil etmesi için(son indisleri de alması için)


text = pytesseract.image_to_string(kirp, lang='eng', config='--psm 8 --oem 3')
text=text.strip().replace(" ","")
print("Plaka: ", text)

cv2.imshow("Gri", gri)
#cv2.imshow("Filtre", filtre)
#cv2.imshow("Kose", kose)
#cv2.imshow("Maske", yeni_maske)
cv2.imshow("Yazi", yazi)
cv2.imshow(f"{text}", kirp)


cv2.waitKey(0)
cv2.destroyAllWindows()




