import subprocess
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image, AsyncImage
from kivy.uix.carousel import Carousel
from kivy.base import EventLoop
import cv2
import numpy as np
import csv

class menuApp(App):
    def built(self):
        self.out = []

class startButton(Button):
    def on_release(self):
        files = self.parent.out[0]
        if len(files) == 1:
            global aug
            aug = self.parent.out
            EventLoop.exit()

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        global flag
        global points
        flag = True
        points.append([x, y])

aug = []
points = []
menuApp().run()
file_path = aug[0][0].split('/')[-1]
height = aug[1]
width = aug[2]
print("file path : ", file_path)
print("height : {}, width : {}".format(height, width))

#field_image = cv2.imread('field.png')
#field_image = cv2.resize(field_image, dsize=(height*10, width*10))

video = cv2.VideoCapture(file_path)
ret, frame = video.read()
flag = False

cv2.namedWindow("point", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("point", onMouse)

for i in range(1, 17):
    file_name = "field" + str(i).zfill(2) + ".png"
    field_image = cv2.imread(file_name, -1)
    a = field_image[:, :, 3]
    idx = np.where(a == 0)
    field_image[idx[0], idx[1], 0:3] = 0

    #field_image = cv2.resize(field_image, dsize=(height * 10, width * 10))
    while True:
        cv2.imshow("point", frame)
        cv2.imshow("field", field_image)
        #cv2.imshow("idx", idx)
        cv2.waitKey(1)
        if flag:
            break
    flag = False

cv2.destroyAllWindows()

#print("points")
#print(points)

csv_name = './' + file_path.split('.')[0] + '.csv'

with open(csv_name, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(points)

process_mes = "python main.py --source ./" + file_path

subprocess.run(process_mes, shell=True)