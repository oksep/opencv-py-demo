import cv2
import numpy as np


def show_img(mat):
    cv2.imshow('img', mat)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


# 读取图像
def read_img():
    mat = cv2.imread('./assets/cat_a.png', cv2.IMREAD_COLOR)
    show_img(mat)


# 保存图像
def write_img():
    mat = cv2.imread('./assets/cat_a.png', cv2.IMREAD_COLOR)
    cv2.imwrite('./assets/cat_a_copy.png', mat)


# 读取摄像头做高斯模糊

def show_camera():
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('video', 640, 480)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # frame = cv2.flip(frame, 1)
        frame = cv2.GaussianBlur(frame, (65, 65), 50)

        cv2.imshow('video', frame)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break

    cap.release()


# ROI (region of interest)
def show_roi():
    mat = cv2.imread('./assets/cat_a.png', cv2.IMREAD_COLOR)
    roi = mat[0:500, 0:500, :]
    show_img(roi)


# 播放视频并在每一帧上绘制文本
def play_video():
    from datetime import datetime

    cap = cv2.VideoCapture('./assets/SampleVideo_1280x720_1mb.mp4')

    while True:

        # Capture frames in the video
        ret, frame = cap.read()

        # describe the type of font
        # to be used.
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Use putText() method for
        # inserting text on video
        cv2.putText(
            frame,
            'TEXT ON VIDEO' + datetime.now().strftime("%H:%M:%S"),
            (50, 50),
            font,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_4
        )

        size = frame.size

        # Display the resulting frame
        cv2.imshow('video', frame)

        # creating 'q' as the quit
        # button for the video
        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    # release the cap object
    cap.release()
    # close all windows
    cv2.destroyAllWindows()


def bite_operation():
    mat = cv2.imread('./assets/cat_a.png', cv2.IMREAD_COLOR)
    h, w, c = mat.shape
    show_img(mat)

    text = cv2.imread('./assets/text.png', cv2.IMREAD_COLOR)
    text = cv2.resize(text, (h, w))

    mask = cv2.imread('./assets/mask.png', cv2.IMREAD_COLOR) # np.zeros((h, w, c), np.uint8)
    mask = cv2.resize(mask, (h, w))

    mask[mask < 127] = 0
    mask[mask >= 127] = 255

    # mask = np.zeros((h, w), np.uint8)
    # mask[100:400, 200:400] = 255
    # mask[100:500, 100: 200] = 255
    #
    # mask[100:400, 200:400] = 1
    # mask[100:500, 100: 200] = 1
    # mat = mat * mask

    show_img(mask)

    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
    show_img(mask)

    # show_img(cv2.add(mat, text, mask=mask))


def hsv_skin_range():
    mat = cv2.imread('./assets/girl.png', cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(mat, cv2.COLOR_BGR2HSV)
    show_img(mat)
    show_img(hsv)

    # Hue: [0 - 33]
    # Saturation: [10 - 255]
    # Value: [80 - 255]
    min_hsv = np.array([0, 10, 80], np.uint8)
    max_hsv = np.array((33, 255, 255), np.uint8)
    mask = cv2.inRange(hsv, min_hsv, max_hsv)

    show_img(mask)

    show_img(cv2.bitwise_and(mat, mat, mask=mask))


hsv_skin_range()
