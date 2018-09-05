# encoding:utf8


import numpy as np
import cv2


def getPerspectiveTransformByAngle(anglex, angley, anglez, distance_angle, width, height):
    """
        通过旋转角度获取图像的透视参数

        镜头位于z轴上，即我们所看到的是由x轴（水平）和y轴（垂直）组成的二维平面，图像就位于该二维平面上，垂直于z轴，
        且图像中心位于z轴的0原点

        :param anglex           绕x轴旋转的的角度
        :param angley           绕y轴旋转的的角度
        :param anglez           绕z轴旋转的的角度，这里正数为逆时针，负数为顺时针
        :param distance_angle   镜头到图像顶角的连线与z轴的夹角，所以角度越小，镜头距离图像越远，范围(0,90)开区间
        :param height           图像的高度
        :param width            图像的宽度

        当anglex和angley均为0时，图像位于x和y轴组成的二维平面内，无论镜头距离原点多远，看的效果都一样
        只有当anglex和angley不均为0时，图像才会离开x-y二维平面，此时镜头距离就有效果了，而且可知：
        当distance_angle<45度时，无论怎样透视，都能看到图像全貌
        当distance_angle>45度时，可能会导致无法看到图像全貌

    """

    def rad(x):
        return x * np.pi / 180

    z = np.sqrt(width ** 2 + height ** 2) / 2 / np.tan(rad(distance_angle))  # 镜头与图像间的距离

    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    r = rx.dot(ry).dot(rz)

    center = np.array([width / 2, height / 2, 0, 0], np.float32)
    src1 = np.array([0, 0, 0, 0], np.float32) - center
    src2 = np.array([width, 0, 0, 0], np.float32) - center
    src3 = np.array([0, height, 0, 0], np.float32) - center
    src4 = np.array([width, height, 0, 0], np.float32) - center

    list_dst = [r.dot(src1), r.dot(src2), r.dot(src3), r.dot(src4)]
    src = np.array([[0, 0], [width, 0], [0, height], [width, height]], np.float32)
    dst = np.zeros((4, 2), np.float32)

    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + center[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + center[1]

    factors = cv2.getPerspectiveTransform(src, dst)

    return factors


img1 = cv2.imread("2.jpg", 1)
border = 400
img1 = cv2.copyMakeBorder(img1, border, border, border, border, cv2.BORDER_CONSTANT, 0)

anglex = 0
angley = 0
anglez = 0
distance_angle = 30
height, width = img1.shape[0:2]

while True:
    factors = getPerspectiveTransformByAngle(anglex, angley, anglez, distance_angle, width, height)
    img = cv2.warpPerspective(img1, factors, (width, height))
    cv2.imshow("test1", img)
    c = cv2.waitKey(30)  # 返回一个按键字符的ASCII编码，这里用键盘控制透视效果
    if c == ord('w'):
        anglex += 1
    elif c == ord('s'):
        anglex -= 1
    elif c == ord('d'):
        angley += 1
    elif c == ord('a'):
        angley -= 1
    elif c == ord('u'):
        anglez += 1
    elif c == ord('i'):
        anglez -= 1
    elif c == ord('z'):
        anglex = angley = anglez = 0
    elif c == ord('q'):
        break
