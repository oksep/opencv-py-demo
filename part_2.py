import cv2
import numpy as np


def show_img(mat, window_name=None):
    cv2.imshow(window_name if (window_name) else 'img', mat)
    print(window_name if (window_name) else 'img')
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


# Encryption and decryption

def encryption_decryption():
    mat = cv2.imread('./assets/cat_a.png')
    show_img(mat)

    w, h, c = mat.shape
    key = np.random.randint(0, 256, (w, h, c), np.uint8)
    show_img(key)

    encryption = cv2.bitwise_xor(mat, key)
    show_img(encryption)

    decryption = cv2.bitwise_xor(encryption, key)
    show_img(decryption)


def encryption_decryption_mask():
    # 知识点
    # mask = np.zeros((3, 3), np.uint8)
    # print(mask)
    # mask[0:2,0:2] = 1
    # print(mask)
    # mask = 1 - mask
    # print(mask * 255)

    mat = cv2.imread('./assets/cat_a.png')

    w, h, c = mat.shape
    key = np.random.randint(0, 256, (w, h, c), np.uint8)

    # 原图加密
    cat_xor_key = cv2.bitwise_xor(mat, key)
    show_img(cat_xor_key, '原图加密')

    # 面部掩码
    mask = np.zeros((w, h, c), np.uint8)
    mask[277: (277 + 438), 210: (210 + 522)] = 1
    show_img(mask * 255, '面部掩码')

    # 加密面部
    encrypt_face = cv2.bitwise_and(cat_xor_key, mask * 255)
    show_img(encrypt_face, '加密面部')

    # 去除面部抠图
    no_face1 = cv2.bitwise_and(mat, (1 - mask) * 255)
    show_img(no_face1, '去除面部抠图')

    # 原图 + 面部加密
    mask_face = encrypt_face + no_face1
    show_img(mask_face, '原图 + 面部加密')

    # (原图 + 面部加密)的解密
    extract_origin = cv2.bitwise_xor(mask_face, key)
    show_img(extract_origin, '(原图 + 面部加密)的解密')

    # 面部解密
    extract_face = cv2.bitwise_and(extract_origin, mask * 255)
    show_img(extract_face, '面部解密')

    # 去除面部抠图
    no_face2 = cv2.bitwise_and(mask_face, (1 - mask) * 255)
    show_img(no_face2, '去除面部抠图')

    # 原图
    extract_cat = extract_face + no_face2
    show_img(extract_cat, '原图')


# mask[277: (277 + 438), 210: (210 + 522)] = 1
# show_img(mask * 255)
# print(mask)
#
# # mask = cv2.bitwise_and(mat, mask)
# # mask = cv2.bitwise_or(mat, mask)
#
# show_img((1-mask) * 255)
# print(mask)

def encryption_decryption_roi():
    mat = cv2.imread('./assets/cat_a.png')
    roi = mat[277: (277 + 438), 210: (210 + 522)]
    show_img(roi, 'roi')

    w, h, c = roi.shape
    key = np.random.randint(0, 256, (w, h, c), np.uint8)

    secret_face = cv2.bitwise_xor(roi, key)
    show_img(secret_face, 'secret_face')

    mat[277: (277 + 438), 210: (210 + 522)] = secret_face
    show_img(mat, 'mat')

    extract_face = cv2.bitwise_xor(mat[277: (277 + 438), 210: (210 + 522)], key)
    show_img(extract_face, 'extract_face')

    mat[277: (277 + 438), 210: (210 + 522)] = extract_face
    show_img(mat, 'mat')


def bit_plane():
    mat = cv2.imread('./assets/cat_a.png', cv2.IMREAD_GRAYSCALE)
    show_img(mat)
    r, c = mat.shape
    x = np.zeros((r, c, 8), dtype=np.uint8)
    for i in range(8):
        x[:, :, i] = 2 ** i
    ri = np.zeros((r, c, 8), dtype=np.uint8)
    for i in range(8):
        ri[:, :, i] = cv2.bitwise_and(mat, x[:, :, i])
        mask = ri[:, :, i] > 0
        ri[mask] = 255
        show_img(ri[:, :, i], str(i))


def digital_water():
    mat = cv2.imread('./assets/cat_a.png', cv2.IMREAD_GRAYSCALE)
    r, c = mat.shape

    # =========== 嵌入过程 ===========
    # 254 = 二进制(1111 1110) 最低位0
    t1 = np.ones((r, c), dtype=np.uint8) * 254
    # 最低有效位 Least Significant Bit
    lsb0 = cv2.bitwise_and(mat, t1)
    # 需要嵌入的图片
    w = cv2.imread('./assets/text.png', cv2.IMREAD_GRAYSCALE)
    wc = w.copy()
    wc[wc > 0] = 1
    # 在最低位嵌入
    wo = cv2.bitwise_or(lsb0, wc)
    show_img(w)
    show_img(wc)
    show_img(wo)

    # =========== 提取过程 ===========
    t2 = np.ones((r, c), dtype=np.uint8)
    ewb = cv2.bitwise_and(wo, t2)
    ewb[ewb > 0] = 255
    show_img(ewb)


def vision_digital_water():
    A = cv2.imread('./assets/cat_a.png')
    B = cv2.imread('./assets/text.png', cv2.IMREAD_GRAYSCALE)

    B = 255 - B
    C = B.copy()
    w = C[:, :] > 0
    C[w] = 1
    A[:, :, 0] = A[:, :, 0] * C
    A[:, :, 1] = A[:, :, 1] * C
    A[:, :, 2] = A[:, :, 2] * C

    show_img(A)


def vision_digital_water_add():
    A = cv2.imread('./assets/cat_a.png', cv2.IMREAD_GRAYSCALE)
    B = cv2.imread('./assets/text.png', cv2.IMREAD_GRAYSCALE)
    B[B > 0] = 255

    # 溢出取 模运算
    add1 = A + B
    # 溢出取 255
    add2 = cv2.add(A, B)
    # 按权重累加
    add3 = cv2.addWeighted(A, 0.6, B, 0.4, 0)

    show_img(B)
    show_img(add1)
    show_img(add2)
    show_img(add3)


def vision_digital_water_art_text():
    A = cv2.imread('./assets/lena_color.png')
    B = cv2.imread('./assets/digital_mask.bmp')
    show_img(cv2.bitwise_or(A, B))


# vision_digital_water_art_text()

# B = cv2.imread('./assets/cat_d.png', cv2.IMREAD_GRAYSCALE)
# _, b = cv2.threshold(B, 127, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# r, c = B.shape
# B = np.zeros((r, c), np.uint8) + 255
# b2 = cv2.drawContours(B, contours[0], 0, (0, 0, 255), 3)
# b3 = cv2.drawContours(B, contours[0], 3, (0, 0, 255), 3)
#
# # print(contours)
#
# print(hierarchy)
#
# # m = cv2.moments(B)
#
# # show_img(b)
# #
# # show_img(b2)
# show_img(b3)

# cv = cv2
# im = cv.imread('./assets/test.png')
# assert im is not None, "file could not be read, check with os.path.exists()"
# imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(imgray, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
# cnt = contours[len(contours) - 1]
# s = cv.drawContours(imgray, [cnt], 0, (0, 255, 106), 3)
# # show_img(s)
#
#
# a = cv.getStructuringElement(cv.MORPH_ELLIPSE, (512, 512))
#
# a[a > 0] = 255
#
# show_img(a)


def Filter(contours):
    sum = 0
    a = []
    for i, area in enumerate(contours):
        if cv2.contourArea(area) > 800:
            a.append(contours[i])
            sum += 1
    return sum, a


def a():
    frame = cv2.imread("./assets/obj_count.png")
    gray = 255 - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将图片变为灰度图片
    ret2, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 算法自动找出合适阈值ret2，将灰度图转换为黑白图，thresh2为返回的黑白图

    contours, hirearchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找出连通域
    sum, contours_1 = Filter(contours)
    img1 = cv2.drawContours(frame, contours_1, -1, (0, 127, 255), 3)

    # 所有操作结束后进行释放
    cv2.imshow("img1", gray)
    cv2.imshow("img2", thresh2)
    cv2.imshow("img3", img1)
    print(sum)
    if cv2.waitKey():
        cv2.destroyAllWindows()


image0 = cv2.imread('./assets/cat_a.png')
image = cv2.resize(image0, (640, 500))
cv2.imshow('original', image)
cv2.waitKey(0)
'''图片预处理'''
H, W = image.shape[:2]  # 获取尺寸
blod = cv2.dnn.blobFromImage(image, 1, (H, W), (0, 0, 0), swapRB=True, crop=False)
net = cv2.dnn.readNet('./assets/starry_night.t7')
# net=cv2.dnn.readNet('./mosaic.t7')
# net=cv2.dnn.readNet('./the_scream.t7')
# net=cv2.dnn.readNet('./the_wave.t7')
# net=cv2.dnn.readNet('./udnie.t7')
# net=cv2.dnn.readNet('feathers.t7')
# net=cv2.dnn.readNet('./candy.t7')
# net=cv2.dnn.readNet('./composition_vii.t7')
# net=cv2.dnn.readNet('./la_muse.t7')
net.setInput(blod)
out = net.forward()
out_new = out.reshape(out.shape[1], out.shape[2], out.shape[3])  # 将输出进行加一化处理
cv2.normalize(out_new, out_new, norm_type=cv2.NORM_MINMAX)
result = out_new.transpose(1, 2, 0)  # 通道转换
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
