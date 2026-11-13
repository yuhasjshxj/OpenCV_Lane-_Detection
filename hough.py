import cv2
import numpy as np

img = cv2.imread("images/lane_test.jpg")
if img is None:
    raise SystemExit("이미지를 못 찾았습니다: images/lane_test.jpg")

scale = 640 / img.shape[1]
img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
h, w = img.shape[:2]

# 2~5단계
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

roi_pts = np.array([[
    (int(0.00 * w), int(0.92 * h)),
    (int(0.40 * w), int(0.52 * h)),
    (int(0.58 * w), int(0.52 * h)),
    (int(0.90 * w), int(0.92 * h)),
]], dtype=np.int32)
mask = np.zeros_like(edges)
cv2.fillPoly(mask, roi_pts, 255)
roi_edges = cv2.bitwise_and(edges, mask)

# 6단계: HoughLinesP (임시 파라미터, 640x360 기준)
lines = cv2.HoughLinesP(
    roi_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=20,
    minLineLength=30,
    maxLineGap=20,
)

overlay = img.copy()
rows = []
if lines is None:
    print("선분을 찾지 못했습니다")
else:
    lines = lines.reshape(-1, 4)          # (N,1,4) 와 (N,4) 모두 대응
    print("검출된 선분 개수:", len(lines))
    print("lines 모양:", lines.shape)

    for x1, y1, x2, y2 in lines:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue                      # 수직선은 기울기 계산 불가, 일단 잠시 제외
        slope = dy / dx                   # 이미지 좌표 기준 (y가 아래로 증가임)
        length = float(np.hypot(dx, dy))
        color = (255, 0, 0) if slope < 0 else (0, 0, 255)   # 파랑=음수, 빨강=양수
        cv2.line(overlay, (x1, y1), (x2, y2), color, 2)
        rows.append((slope, length, x1, y1, x2, y2))

    rows.sort(key=lambda r: r[0])
    print("기울기   길이   (x1,y1) -> (x2,y2)")
    for slope, length, x1, y1, x2, y2 in rows:
        print(f"{slope:+.2f}  {length:6.1f}  ({x1},{y1}) -> ({x2},{y2})")

cv2.imwrite("output/7_hough.jpg", overlay)
print("저장 완료: output/7_hough.jpg")
 
