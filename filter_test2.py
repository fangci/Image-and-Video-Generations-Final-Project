import cv2
import numpy as np

# 讀取黑底白線圖
img = cv2.imread('/home/moony/storage/fangci/AnimateDiff/__assets__/demos/scribble/forest.png', cv2.IMREAD_GRAYSCALE)

# 二值化
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 膨脹線條
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 調整(3,3)改變粗細
dilated = cv2.dilate(thresh, kernel, iterations=2)
dilated = cv2.medianBlur(dilated, 25)
hp_kernal = np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]], dtype=np.float32)
high_pass = cv2.filter2D(dilated, ddepth=-1, kernel=hp_kernal)
high_pass_image = cv2.convertScaleAbs(high_pass)


# 儲存
cv2.imwrite('/home/moony/storage/fangci/AnimateDiff/__assets__/demos/scribble/forest_contour2.png', high_pass_image)