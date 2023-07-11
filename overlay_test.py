import cv2


def overlay(fore_img, back_img, shift):
    '''
    fore_img：合成する画像
    back_img：背景画像
    shift：左上を原点としたときの移動量(x, y)
    '''

    shift_x, shift_y = shift

    fore_h, fore_w = fore_img.shape[:2]
    fore_x_min, fore_x_max = 0, fore_w
    fore_y_min, fore_y_max = 0, fore_h

    back_h, back_w = back_img.shape[:2]
    back_x_min, back_x_max = shift_y, shift_y + fore_h
    back_y_min, back_y_max = shift_x, shift_x + fore_w

    if back_x_min < 0:
        fore_x_min = fore_x_min - back_x_min
        back_x_min = 0

    if back_x_max > back_w:
        fore_x_max = fore_x_max - (back_x_max - back_w)
        back_x_max = back_w

    if back_y_min < 0:
        fore_y_min = fore_y_min - back_y_min
        back_y_min = 0

    if back_y_max > back_h:
        fore_y_max = fore_y_max - (back_y_max - back_h)
        back_y_max = back_h

    back_img[back_y_min:back_y_max, back_x_min:back_x_max] = fore_img[fore_y_min:fore_y_max, fore_x_min:fore_x_max]

    return back_img


fore_img = cv2.imread("ball.png")
back_img = cv2.imread("field.png")

fore_img = cv2.resize(fore_img, dsize=(80, 100))

back_img = overlay(fore_img, back_img, (10, 120))

cv2.imshow('img', back_img)
cv2.waitKey(0)
cv2.destroyAllWindows()