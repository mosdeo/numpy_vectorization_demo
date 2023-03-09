import numpy as np
import cv2 as cv
import time
# Python 版本: 3.11
# NumPy 版本: 1.24.1
# 日期: 2023-3-8
# 作者: Lin Kao-Yuan 林高遠
# 知乎: www.zhihu.com/people/lin-kao-yuan
# 網站: web.ntnu.edu.tw/~60132057A
# Github: github.com/mosdeo

# 需求：把圖片中的膚色去除，只留下其他顏色
# 注意：這份程式碼只是示範如何使用向量化，並不是在教學如何做膚色檢測，不要來跟我槓膚色檢測效果好壞

folder = "03_color_filter_vectorlize/"
img = cv.imread(folder + "boys.jpg")
lower_skin = [0, 48, 80]
upper_skin = [18, 225, 255]

# 方法1: 使用 for 迴圈 (最慢)
img_forloop = img.copy()
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) # 轉換成 HSV
h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2] # 分離出 H, S, V
start_time = time.time()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if h[i, j] > lower_skin[0] and h[i, j] < upper_skin[0] and \
            s[i, j] > lower_skin[1] and s[i, j] < upper_skin[1] and \
            v[i, j] > lower_skin[2] and v[i, j] < upper_skin[2]:
            img_forloop[i, j] = 0
elapsed_time_for_loop = time.time() - start_time
print("Elapsed time for loop: %s seconds" % (elapsed_time_for_loop))
cv.imwrite(folder + "for_loop.jpg", img_forloop)

# 方法2: 使用 NumPy 向量化
img_vectorlize = img.copy()
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) # 轉換成 HSV
h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2] # 分離出 H, S, V
start_time = time.time()
skin_mask = np.bitwise_and.reduce([
        h > lower_skin[0], h < upper_skin[0],
        s > lower_skin[1], s < upper_skin[1],
        v > lower_skin[2], v < upper_skin[2]
    ])
# 給綠色
img_vectorlize[skin_mask] = [0, 196, 0]
elapsed_time_vectorlize = time.time() - start_time
print("Elapsed time vectorlize: %s seconds" % (elapsed_time_vectorlize))
cv.imwrite(folder + "vectorlize.jpg", img_vectorlize)

# 校驗
print("以 np.array_equal 檢查兩種方法的結果是否相同? %s" % np.array_equal(img_forloop, img_vectorlize))

# 比較兩種方法的執行時間
print("Speed up: %.2f" % (elapsed_time_for_loop / elapsed_time_vectorlize))