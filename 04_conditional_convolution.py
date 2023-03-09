import numpy as np
import cv2 as cv
import time
# Python 版本: 3.11
# NumPy 版本: 1.24.1
# 日期: 2023-3-9
# 作者: Lin Kao-Yuan 林高遠
# 知乎: www.zhihu.com/people/lin-kao-yuan
# 網站: web.ntnu.edu.tw/~60132057A
# Github: github.com/mosdeo

# 條件式卷積
# 需求：
# - 訪問每一個像素點，根據周圍的像素點的中位數，來決定自己的值
# - 如果周圍的中位數大於自己就+1，小於則就-1，自己就是中位數的話不變

h, w = 2560, 1300

# 生成 h * w 隨機內容的 uint8 陣列
def get_map():
    np.random.seed(0) # 固定隨機種子，讓每次執行的結果都一樣，方便大家比較
    the_map = np.random.randint(0, 255, size=(h, w), dtype=np.uint8)
    return the_map

# 方法1: 使用 for 迴圈 (最慢)
def for_loop():
    map_prev = get_map()
    map_prev = np.pad(map_prev, 1, 'constant', constant_values=0) # 處理邊界問題，在原本的陣列外面包一圈 0
    my_map_forloop = get_map()
    for y in range(h):
        for x in range(w):
            # 如果周圍的中位數大於自己就+1，小於則就-1，自己就是中位數的話不變
            around9 = map_prev[y:y+3, x:x+3].flatten()
            median = np.median(around9)
            if median > my_map_forloop[y, x]:
                my_map_forloop[y, x] += 1
            elif median < my_map_forloop[y, x]:
                my_map_forloop[y, x] -= 1
    return my_map_forloop

# 方法2: 使用 NumPy 向量化
def numpy_vectorization():
    my_map_vectorlize = get_map()
    around9_of_my_map = np.zeros((h+2, w+2, 9), dtype=np.uint8) #+2是為了避免邊界問題
    # 從左上角開始，順時針方向，旋轉梯式整片賦值
    # 作者: Lin Kao-Yuan 林高遠
    # 知乎: www.zhihu.com/people/lin-kao-yuan
    # 網站: web.ntnu.edu.tw/~60132057A
    # around9_of_my_map[Y位置, X位置, 第幾個方向]
    around9_of_my_map[0:-2, 0:-2, 0] = my_map_vectorlize
    around9_of_my_map[0:-2, 1:-1, 1] = my_map_vectorlize
    around9_of_my_map[0:-2, 2:  , 2] = my_map_vectorlize
    around9_of_my_map[1:-1, 0:-2, 3] = my_map_vectorlize
    around9_of_my_map[1:-1, 1:-1, 4] = my_map_vectorlize # 自己
    around9_of_my_map[1:-1, 2:  , 5] = my_map_vectorlize
    around9_of_my_map[2:  , 0:-2, 6] = my_map_vectorlize
    around9_of_my_map[2:  , 1:-1, 7] = my_map_vectorlize
    around9_of_my_map[2:  , 2:  , 8] = my_map_vectorlize
    
    # 裁掉邊界
    around9_of_my_map = around9_of_my_map[1:-1, 1:-1, :]

    # 計算中位數
    median_map = np.median(around9_of_my_map, axis=2)[1:-1, 1:-1].astype(np.uint8)

    # 比較中位數和自己的大小
    # 如果周圍的中位數大於自己就+1，小於則就-1，自己就是中位數的話不變
    my_map_vectorlize = np.where(median_map > my_map_vectorlize, my_map_vectorlize + 1, my_map_vectorlize)
    my_map_vectorlize = np.where(median_map < my_map_vectorlize, my_map_vectorlize - 1, my_map_vectorlize)
    return my_map_vectorlize

if __name__ == '__main__':
    start_time = time.time()
    my_map_forloop = for_loop()
    elapsed_time_forloop = time.time() - start_time
    print("for_loop time:", elapsed_time_forloop)

    start_time = time.time()
    my_map_vectorlize = numpy_vectorization()
    elapsed_time_vectorlize = time.time() - start_time
    print("numpy_vectorization time:", elapsed_time_vectorlize)

    # 檢驗兩個方法的結果是否一樣
    print("Are the two methods the same? %s" % (np.all(my_map_forloop == my_map_vectorlize)))

    # 檢驗兩個方法的相似度
    print("Similarity: %s" % (np.sum(my_map_forloop == my_map_vectorlize) / (h * w)))

    # 比較兩種方法的執行時間
    print("Speed up: %s" % (elapsed_time_forloop / elapsed_time_vectorlize))

    # 檢驗兩個方法的相似度(忽略邊界)
    # print("Similarity(ignore border): %s" % (np.sum(my_map_forloop[1:-1, 1:-1] == my_map_vectorlize[1:-1, 1:-1]) / ((h-2) * (w-2))))

    # # 印出有差異的元素對比
    # diff_yx = np.argwhere(my_map_forloop != my_map_vectorlize)
    # for y, x in diff_yx:
    #     print("For loop: %s, Vectorlize: %s" % (my_map_forloop[y, x], my_map_vectorlize[y, x]))

    # # 放大圖片100倍
    # diff_is_black = (my_map_forloop == my_map_vectorlize).astype(np.uint8) * 255
    # diff_is_black = cv.resize(diff_is_black, (w * 100, h * 100), interpolation=cv.INTER_NEAREST)
    # my_map_forloop = cv.resize(my_map_forloop, (w * 100, h * 100), interpolation=cv.INTER_NEAREST)
    # my_map_vectorlize = cv.resize(my_map_vectorlize, (w * 100, h * 100), interpolation=cv.INTER_NEAREST)
    # cv.imshow("diff is black", diff_is_black)
    # cv.imshow("my_map_forloop", my_map_forloop)
    # cv.imshow("my_map_vectorlize", my_map_vectorlize)
    # cv.waitKey(0)