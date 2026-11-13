import cv2

img = cv2.imread("images/lane_test.jpg")
if img is None:
    raise SystemExit("이미지를 못 찾았습니다: images/lane_test.jpg")

# 전송 속도를 위해 화면 640으로 축소
scale = 640 / img.shape[1]
img = cv2.resize(img, None)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # 2. 그레이스케일
blur = cv2.GaussianBlur(gray, (5, 5), 0)            # 3. 가우시안 블러(저역통과)
edges = cv2.Canny(blur, 50, 150)                    # 4. Canny

cv2.imwrite("output/0_resized.jpg", img)
cv2.imwrite("output/1_gray.jpg", gray)
cv2.imwrite("output/2_blur.jpg", blur)
cv2.imwrite("output/3_edges.jpg", edges)
print("저장 완료: output/ 폴더 확인")
