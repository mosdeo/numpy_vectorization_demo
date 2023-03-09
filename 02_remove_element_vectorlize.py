import numpy as np
import time
# Python 版本: 3.11
# NumPy 版本: 1.24.1
# 日期: 2023-3-8
# 作者: Lin Kao-Yuan 林高遠
# 知乎: www.zhihu.com/people/lin-kao-yuan
# 網站: web.ntnu.edu.tw/~60132057A
# Github: github.com/mosdeo

# 中國有 3000 萬剩男，所以我隨機產生 3000 萬個身高數據
np.random.seed(0) # 固定隨機種子，讓每次執行的結果都一樣，方便大家比較
heights = np.random.randint(150, 200, 30000000) # 上下界設定為 150~200 cm

# 需求: 找出身高大於 178 cm 的資料
# 因為我的身高是 178 cm，所以我想抓出身高比我能吸引的女人的樣本

# 使用 for 迴圈 (最慢)
start_time = time.time()
heights_above_178 = []
for height in heights:
    if height > 178:
        heights_above_178.append(height)
elapsed_time_for_loop = time.time() - start_time
print("檢驗結果: 高於178的人數為 %s" % (len(heights_above_178)))

# 使用向量化 (最快)
start_time = time.time()
heights_above_178 = heights[heights > 178]
elapsed_time_vectorlize = time.time() - start_time
print("檢驗結果: 高於178的人數為 %s" % (len(heights_above_178)))

# 比較兩種方法的執行時間
print("Elapsed time for loop:   %s seconds" % (elapsed_time_for_loop))
print("Elapsed time vectorlize: %s seconds" % (elapsed_time_vectorlize))
print("Speed up: %.2f" % (elapsed_time_for_loop / elapsed_time_vectorlize))