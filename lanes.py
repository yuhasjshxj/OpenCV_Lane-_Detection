import cv2
import numpy as np

# @@@@@@@(이 이미지 기준, 카메라 확정 후 재조정)@@@@@@@
LEFT_SLOPE = (-1.5, -0.3)    # 왼쪽 차선: 기울기 음수
RIGHT_SLOPE = (0.8, 1.5)     # 오른쪽 차선: 기울기 양수 (0.65 근처 대각선은 제외)

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

# 5단계
y_top, y_bot = int(0.52 * h), int(0.92 * h)
roi_pts = np.array([[
    (int(0.00 * w), y_bot),
    (int(0.40 * w), y_top),
    (int(0.58 * w), y_top),
    (int(0.90 * w), y_bot),
]], dtype=np.int32)
mask = np.zeros_like(edges)
cv2.fillPoly(mask, roi_pts, 255)
roi_edges = cv2.bitwise_and(edges, mask)

# 6단계
lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180,
                        threshold=20, minLineLength=30, maxLineGap=20)
if lines is None:
    raise SystemExit("선분을 찾지 못했습니다")
lines = lines.reshape(-1, 4)

# 7단계 기울기로 좌/우 분류
left_segs, right_segs, excluded = [], [], []
for x1, y1, x2, y2 in lines:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    seg = ((x1, y1), (x2, y2))
    if x2 == x1:
        excluded.append(seg)
        continue
    slope = (y2 - y1) / (x2 - x1)
    if LEFT_SLOPE[0] < slope < LEFT_SLOPE[1]:
        left_segs.append(seg)
    elif RIGHT_SLOPE[0] < slope < RIGHT_SLOPE[1]:
        right_segs.append(seg)
    else:
        excluded.append(seg)


def fit_lane(segments, y_top, y_bot):
    """선분 그룹의 끝점을 모아 x = a*y + c 직선 하나로 맞추고,
    y_top, y_bot 높이에서의 두 점을 돌려줍니다. 선분이 없으면 None."""
    if not segments:
        return None
    ys = [p[1] for seg in segments for p in seg]
    xs = [p[0] for seg in segments for p in seg]
    a, c = np.polyfit(ys, xs, 1)
    return (int(round(a * y_top + c)), y_top), (int(round(a * y_bot + c)), y_bot)


left = fit_lane(left_segs, y_top, y_bot)
right = fit_lane(right_segs, y_top, y_bot)

print(f"왼쪽 선분 {len(left_segs)}개 / 오른쪽 선분 {len(right_segs)}개 / 제외 {len(excluded)}개")
print("왼쪽 대표선 (위, 아래):", left)
print("오른쪽 대표선 (위, 아래):", right)
if left and right:
    print("차선 상태: 정상")
elif left:
    print("차선 상태: 오른쪽 놓침")
elif right:
    print("차선 상태: 왼쪽 놓침")
else:
    print("차선 상태: 둘 다 놓침")

# 제외된 선분은 회색, 대표선은 초록색이다
overlay = img.copy()
for p1, p2 in excluded:
    cv2.line(overlay, p1, p2, (160, 160, 160), 2)
for lane in (left, right):
    if lane:
        cv2.line(overlay, lane[0], lane[1], (0, 255, 0), 3)

cv2.imwrite("output/8_lanes.jpg", overlay)
print("저장 완료: output/8_lanes.jpg")
