import cv2
import numpy as np

img = cv2.imread("images/lane_test.jpg")
if img is None:
    raise SystemExit("이미지를 못 찾았습니다: images/lane_test.jpg")

scale = 640 / img.shape[1]
img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
h, w = img.shape[:2]

# 2~4단계
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# 5단계: ROI 사다리꼴
roi_pts = np.array([[
    (int(0.00 * w), int(0.92 * h)),   # 왼쪽 아래 (보닛 위쪽까지임)
    (int(0.40 * w), int(0.52 * h)),   # 왼쪽 위
    (int(0.58 * w), int(0.52 * h)),   # 오른쪽 위
    (int(0.90 * w), int(0.92 * h)),   # 오른쪽 아래
]], dtype=np.int32)

mask = np.zeros_like(edges)
cv2.fillPoly(mask, roi_pts, 255)
roi_edges = cv2.bitwise_and(edges, mask)

# 확인용: 원본 위에 ROI 윤곽선을 그림
overlay = img.copy()
cv2.polylines(overlay, roi_pts, True, (0, 255, 0), 2)

cv2.imwrite("output/4_roi_mask.jpg", mask)
cv2.imwrite("output/5_roi_edges.jpg", roi_edges)
cv2.imwrite("output/6_roi_overlay.jpg", overlay)
print("저장 완료: output/ 폴더 확인")
