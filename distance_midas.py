import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

model_type = (
    "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
# midas = torch.hub.load("intel-isl/MiDas","MiDas_small")
# midas.to("GPU")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
midas.to("cuda")

midas.eval()

transforms = torch.hub.load("intel-isl/MiDas", "transforms")
transform = transforms.dpt_transform

# カメラの設定
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# マウスイベント用のコールバック関数
def mouse_callback(event, x, y, flags, param):
    imgbatch = transform(img).to("cuda")

    with torch.no_grad():
        predction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            predction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()
        # print(output)

    # 左クリックであれば座標を表示し、距離を計算
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at ({x}, {y})")
        # depth = midas(get_depth=True, input=torch.from_numpy(frame).to(device)).squeeze()
        # distance = output[y, x] / 100  # Midasの出力はcmなので、mに変換するために100で割る
        distance = output[y, x]  # Midasの出力はcmなので、mに変換するために100で割る
        print(f"Distance: {distance:.2f} cm")
        time.sleep(3)


# ウィンドウを生成し、マウスイベントのコールバック関数を登録
cv2.namedWindow("Midas")
cv2.setMouseCallback("Midas", mouse_callback)

while True:
    # while cap.isOpened():
    # カメラからフレームを取得
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to("cuda")

    with torch.no_grad():
        predction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            predction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()
        print(output)

    # depth = output
    # plt.imshow(output)
    # 距離の最小値と最大値を計算
    # min_distance = np.min(output) / 100S
    # max_distance = np.max(output) / 100
    min_distance = np.min(output)
    max_distance = np.max(output)
    print(f"Distance range: {min_distance:.2f} - {max_distance:.2f} cm")

    # 距離マップを可視化
    depth_viz = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

    # ウィンドウに画像を表示
    cv2.imshow("Midas", depth_viz)
    # plt.show()

    # 'q'キーで終了
    if cv2.waitKey(1) == ord("q"):
        break

# 後処理
cap.release()
cv2.destroyAllWindows()
