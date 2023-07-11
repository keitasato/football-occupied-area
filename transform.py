import sys

import numpy as np
import cv2

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

def get_homography(img):
    src = cv2.resize(img, (512, 256))
    theta = np.radians(10)
    t1, t2 = 10, 10
    H = np.array([[np.cos(theta), -np.sin(theta), t1]
                  ,[np.sin(theta), np.cos(theta), t2]
                  ,[0, 0, 1] ])
    print("Homography matrix=¥n", H)

    img_1 = []
    for x in range(4):
        v = random.randint(0, 512)
        u = random.randint(0, 256)
        img_1.append([u, v, 1])
    img_1 = np.array(img_1)
    print("img1", img_1.shape)
    print(img_1)

    img_2 = []
    for row in img_1:
        x2 = H @ row
        x2 = x2.astype('int')
        x2[0] = x2[0]
        x2[1] = x2[1]
        img_2.append(x2)
    img_2 = np.array(img_2)
    print("img2", img_2.shape)
    print(img_2)

    M = H[:2,]
    affine_img = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))

    cv2.imshow("color", affine_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    A = np.zeros((img_1.shape[0]*2, 9))
    for i, (a, b) in enumerate(zip(img_1, img_2)):
        A[i * 2] = np.array([a[0], a[1], 1, 0, 0, 0, -b[0] * a[0], -b[0] * a[1], -b[0]])
        A[i * 2 + 1] = np.array([0, 0, 0, a[0], a[1], 1, -b[1] * a[0], -b[1] * a[1], -b[1]])
    print(A.shape)
    print("A = ", A)

    u, s, vh = np.linalg.svd(A)
    print('¥nSVD result')
    print("shape of u, s, vh: ", u.shape, s.shape, vh.shape)
    min = 8
    print("singular values: ", vh[min])

    Hest = vh[min].reshape((3, 3))
    Hest = Hest / Hest[2, 2]
    print("Estimated Homography matrix\n", Hest)

    sys.exit()

video = cv2.VideoCapture('kanai.mp4')

ret, frame = video.read()
cv2.imshow("point", frame)
cv2.setMouseCallback("point", onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()

field_image = cv2.imread('field.png')
field_image = cv2.resize(field_image, dsize=(640, 480))
cv2.imshow("point", field_image)
cv2.setMouseCallback("point", onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()


sys.exit()

p_original = np.float32([ [0, 701], [1053, 673], [528, 939], [1890, 728] ])
p_trans = np.float32([ [0, 0], [640, 0], [0, 480], [640, 480] ])

pts = np.array([ [1500, 700], [700, 29] ], dtype='float32')
pts = np.array([pts])

M = cv2.getPerspectiveTransform(p_original, p_trans)

pts_trans = cv2.perspectiveTransform(pts, M)
print(pts_trans)

while True:
    ret, frame = video.read()
    if ret:
        frame = cv2.warpPerspective(frame, M,  (640, 480))
        cv2.imshow("tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()