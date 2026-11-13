import cv2
import numpy as np

#@@@@@@임시 파라미터 (이 이미지 기준, 카메라 확정 후 재조정)@@@@@@
LEFT_SLOPE = (-1.5, -0.3)    # 왼쪽 차선 기울기 범위
RIGHT_SLOPE = (0.8, 1.5)     # 오른쪽 차선 기울기 범위
HFOV_DEG = 70.0              # 가로 시야각 가정값 (캘리브레이션 후 초점거리로 교체)


img = cv2.imread("images/lane_test.jpg")
if img is None:
    raise SystemExit("이미지를 못 찾았습니다: images/lane_test.jpg")

scale = 640 / img.shape[1]
img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
h, w = img.shape[:2]
cx = w / 2.0                                   # 차량 중심 = 화면 중앙 (임시 가정)

# 2~4단계: 그레이스케일, 블러, Canny
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# 5단계: ROI
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

# 6단계: HoughLinesP
lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180,
                        threshold=20, minLineLength=30, maxLineGap=20)
if lines is None:
    raise SystemExit("선분을 찾지 못했습니다")
lines = lines.reshape(-1, 4)

# 7단계: 좌/우 분류
left_segs, right_segs = [], []
for x1, y1, x2, y2 in lines:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 == x1:
        continue
    slope = (y2 - y1) / (x2 - x1)
    seg = ((x1, y1), (x2, y2))
    if LEFT_SLOPE[0] < slope < LEFT_SLOPE[1]:
        left_segs.append(seg)
    elif RIGHT_SLOPE[0] < slope < RIGHT_SLOPE[1]:
        right_segs.append(seg)


def fit_lane(segments):
    """끝점들을 모아 x = a*y + c 의 (a, c) 반환. 선분이 없으면 None."""
    if not segments:
        return None
    ys = [p[1] for seg in segments for p in seg]
    xs = [p[0] for seg in segments for p in seg]
    a, c = np.polyfit(ys, xs, 1)
    return float(a), float(c)


def x_at(lane, y):
    return lane[0] * y + lane[1]


left = fit_lane(left_segs)
right = fit_lane(right_segs)
overlay = img.copy()

if left is None or right is None:
    state = "오른쪽 놓침" if left else ("왼쪽 놓침" if right else "둘 다 놓침")
    print("차선 상태:", state, "→ 오프셋/방향 오차 계산 불가 (한쪽 처리는 다음 단계)")
else:
    # 8단계: 중간선
    xl_b, xr_b = x_at(left, y_bot), x_at(right, y_bot)
    xl_t, xr_t = x_at(left, y_top), x_at(right, y_top)
    center_b = (xl_b + xr_b) / 2
    center_t = (xl_t + xr_t) / 2

    # 9단계-1: 오프셋 (아래쪽 한 줄 기준, 차선 반폭으로 정규화)
    half_width = (xr_b - xl_b) / 2
    offset = (center_b - cx) / half_width      # +: 차선 중심이 오른쪽 → 오른쪽 조향
    offset = float(np.clip(offset, -1.0, 1.0))

    # 9단계-2: 방향 오차 (소실점 x 위치 → 각도)
    da = left[0] - right[0]
    if abs(da) < 1e-6:
        heading = 0.0
        vp = None
        print("두 차선이 평행하게 검출되어 소실점을 구하지 못했습니다 (방향 오차 0으로 처리)")
    else:
        y_vp = (right[1] - left[1]) / da
        x_vp = x_at(left, y_vp)
        vp = (x_vp, y_vp)
        f_px = (w / 2.0) / np.tan(np.radians(HFOV_DEG / 2))
        heading = float(np.degrees(np.arctan((x_vp - cx) / f_px)))   # +: 차선이 오른쪽을 향함

    print("차선 상태: 정상")
    print(f"아래쪽 차선 폭: {xr_b - xl_b:.0f}px / 차선 중심 x: {center_b:.1f} / 화면 중심 x: {cx:.0f}")
    if vp:
        print(f"소실점: ({vp[0]:.1f}, {vp[1]:.1f})")
    print(f"오프셋 (정규화, +=오른쪽 조향): {offset:+.3f}")
    print(f"방향 오차 (도, +=오른쪽 조향): {heading:+.2f}")

    # 확인용 그림
    for lane, color in ((left, (0, 255, 0)), (right, (0, 255, 0))):
        cv2.line(overlay, (int(x_at(lane, y_top)), y_top), (int(x_at(lane, y_bot)), y_bot), color, 3)
    cv2.line(overlay, (int(center_t), y_top), (int(center_b), y_bot), (0, 255, 255), 3)   # 중간선(노랑)
    cv2.line(overlay, (int(cx), y_top), (int(cx), y_bot), (0, 0, 255), 2)                  # 차량 중심(빨강)
    if vp and 0 <= vp[1] < h:
        cv2.circle(overlay, (int(vp[0]), int(vp[1])), 6, (255, 0, 255), -1)                # 소실점(자홍)

cv2.imwrite("output/9_result.jpg", overlay)
print("저장 완료: output/9_result.jpg")
