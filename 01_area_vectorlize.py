import numpy as np
import time
# Python 版本: 3.11
# NumPy 版本: 1.24.1
# 日期: 2023-3-5
# 作者: Lin Kao-Yuan 林高遠
# 知乎: www.zhihu.com/people/lin-kao-yuan
# 網站: web.ntnu.edu.tw/~60132057A
# Github: github.com/mosdeo

# 隨機產生 1780 萬個矩形的兩個對角點座標，為什麼用這個數字呢？因為178是我的身高
np.random.seed(0) # 固定隨機種子，讓每次執行的結果都一樣，方便大家比較
rects = np.random.randint(0, 100, (17800000, 2, 2))

# 計算這些矩形的面積，使用 for 迴圈
start_time = time.time()
area = []
for rect in rects:
    area.append((abs(rect[0][0] - rect[0][1]) * abs(rect[1][0] - rect[1][1])))
elapsed_time_for_loop = time.time() - start_time
print("檢驗結果: %s" % (sum(area)))
print("Elapsed time: %s" % (elapsed_time_for_loop))

# 計算這些矩形的面積，使用最簡單的向量化
start_time = time.time()
min_x = np.min(rects[:, 0], axis=1)
max_x = np.max(rects[:, 0], axis=1)
min_y = np.min(rects[:, 1], axis=1)
max_y = np.max(rects[:, 1], axis=1)
area = (max_x - min_x) * (max_y - min_y)
elapsed_time_vectorlize = time.time() - start_time
print("檢驗結果: %s" % (sum(area)))
print("Elapsed time: %s" % (elapsed_time_vectorlize))

# 計算這些矩形的面積，使用np.ptp向量化
start_time = time.time()
area = np.ptp(rects[:, 0], axis=1) * np.ptp(rects[:, 1], axis=1)
elapsed_time_ptp = time.time() - start_time
print("檢驗結果: %s" % (sum(area)))
print("Elapsed time: %s" % (elapsed_time_ptp))

# 計算這些矩形的面積，直接取絕對值相乘
start_time = time.time()
area = np.abs(rects[:, 0, 0] - rects[:, 0, 1]) * np.abs(rects[:, 1, 0] - rects[:, 1, 1])
# 下面這種方法更快一點，但可能會發生 Exception has occurred: _ArrayMemoryError
# area = np.abs(np.diff(rects[:, 0], axis=1)) * np.abs(np.diff(rects[:, 1], axis=1)).flatten()
elapsed_time_diff_abs = time.time() - start_time
print("檢驗結果: %s" % (sum(area)))
print("Elapsed time: %s" % (elapsed_time_diff_abs))
# 計算加速比
acceleration_ratio = elapsed_time_for_loop / np.array([
    elapsed_time_vectorlize, 
    elapsed_time_ptp, 
    elapsed_time_diff_abs])
print("Speed up for vectorlize: %sx" % acceleration_ratio[0])
print("Speed up for ptp-vectorlize: %sx" % acceleration_ratio[1])
print("Speed up for diff-abs-vectorlize: %sx" % acceleration_ratio[2])