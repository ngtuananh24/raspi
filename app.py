import cv2
import numpy as np
from ultralytics import YOLO

# Load mô hình YOLOv8 - Cập nhật đường dẫn phù hợp với Raspberry Pi
model = YOLO("best.pt")  # Đặt file model trong cùng thư mục với script

# Lấy nhãn từ mô hình
id_to_label = model.names

# Định nghĩa ánh xạ nhãn cho biển báo giao thông
labels_map = {
    "DP.135": "Hết mọi lệnh cấm",
    "P.102": "Cấm đi ngược chiều",
    "P.103a": "Cấm xe ô tô",
    "P.103b": "Cấm xe ôtô rẽ phải",
    "P.103c": "Cấm ô tô rẽ trái",
    "P.104": "Cấm xe máy",
    "P.106a": "Cấm xe tải đi vào làn đường trừ những xe nằm trong danh sách ưu tiên theo quy định",
    "P.106b": "Cấm các xe ô tô tải có khối lượng chuyên chở",
    "P.107a": "Cấm xe ô tô khách",
    "P.112": "Cấm người đi bộ qua lại",
    "P.115": "Hạn chế tải trọng toàn bộ xe",
    "P.117": "Hạn chế chiều cao",
    "P.123a": "Cấm rẽ trái",
    "P.123b": "Cấm rẽ phải",
    "P.124a": "Cấm quay đầu xe",
    "P.124b": "Cấm ôtô quay đầu",
    "P.124c": "Cấm rẽ trái và quay đầu xe",
    "P.125": "Cấm các loại xe cơ giới vượt nhau",
    "P.127": "Tốc độ tối đa",
    "P.128": "Đoạn đường không được sử dụng còi",
    "P.130": "Cấm dừng xe và đỗ xe",
    "P.131a": "Cấm các phương tiện giao thông đỗ xe",
    "P.137": "Cấm tất cả các loại xe rẽ trái hay rẽ phải ở phía trước",
    "P.245a": "Biển báo đi chậm",
    "R.301c": "Hướng đi phải theo: Các xe chỉ được rẽ trái",
    "R.301d": "Hướng đi phải theo: Rẽ phải",
    "R.301e": "Hướng đi phải theo: Rẽ trái",
    "R.302a": "Hướng vòng sang phải",
    "R.302b": "Hướng vòng sang trái",
    "R.303": "Nơi giao nhau chạy theo vòng xuyến",
    "R.407a": "Cho phép chủ xe đi thẳng theo chiều mũi tên ký hiệu trên biển báo đường 1 chiều",
    "R.409": "Biển chỉ dẫn điểm quay xe R",
    "R.425": "Chỉ dẫn về cơ sở điều trị bệnh gần đường",
    "R.434": "Bến xe buýt",
    "S.509a": "Thuyết minh biển chính - Chiều cao an toàn",
    "W.201a": "Chỗ ngoặt nguy hiểm vòng bên trái",
    "W.201b": "Chỗ ngoặt nguy hiểm vòng bên phải",
    "W.202a": "Biển báo đường ngoặt liên tiếp trái",
    "W.202b": "Biển báo đường ngoặt liên tiếp phải",
    "W.203b": "Đường hẹp bên trái",
    "W.203c": "Đường hẹp bên phải",
    "W.205a": "Đường giao nhau cùng cấp",
    "W.205b": "Đường giao nhau cùng cấp",
    "W.205d": "Đường giao nhau cùng cấp",
    "W.207a": "Giao nhau với đường không ưu tiên 2 bên",
    "W.207b": "Giao nhau với đường không ưu tiên bên phải",
    "W.207c": "Giao nhau với đường không ưu tiên trái",
    "W.208": "Biển báo giao nhau với đường ưu tiên",
    "W.209": "Giao nhau có tín hiệu đèn",
    "W.210": "Nơi giao nhau có đường sắt",
    "W.219": "Dốc xuống nguy hiểm",
    "W.221b": "Đường có gồ giảm tốc",
    "W.224": "Người đi bộ cắt ngang",
    "W.225": "Trẻ em",
    "W.227": "Công trường",
    "W.233": "Nguy hiểm khác",
    "W.235": "Đường Đôi",
    "W.245a": "Biển cảnh báo ĐI CHẬM"
}


def detect_and_draw(image):
    """Nhận diện và vẽ kết quả"""
    results = model(image)
    detected_names = []

    for r in results:
        img_plot = r.plot()
        for box in r.boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            if conf >= 0.5:
                class_code = id_to_label[cls]
                class_name = labels_map.get(class_code, "Unknown")
                area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                detected_names.append((area, f"{class_code}: {class_name} ({conf:.2f})"))

    detected_names.sort(reverse=True, key=lambda x: x[0])
    detected_names = [name[1] for name in detected_names]

    return img_plot, detected_names


def main():
    # Mở camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera. Vui lòng kiểm tra thiết bị!")
        return

    print("Đã khởi động camera. Đang nhận diện biển báo...")
    print("Nhấn 'q' để thoát.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ camera. Thoát...")
                break

            # Thay đổi kích thước frame cho phù hợp với mô hình
            frame_resized = cv2.resize(frame, (640, 640))

            # Nhận diện biển báo
            result_image, detected_names = detect_and_draw(frame_resized)

            # Hiển thị tên biển báo trên cửa sổ OpenCV
            if detected_names:
                # Vẽ một hình chữ nhật đen để làm nền cho văn bản
                cv2.rectangle(result_image, (10, 10), (630, 30 + 25 * len(detected_names)), (0, 0, 0), -1)

                # Hiển thị tên biển báo trên hình ảnh
                for i, name in enumerate(detected_names):
                    cv2.putText(result_image, name, (15, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # In tên biển báo ra terminal
                print("\nBiển báo phát hiện được:")
                for name in detected_names:
                    print(f"- {name}")

            # Hiển thị kết quả
            cv2.imshow('Nhận diện biển báo giao thông', result_image)

            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Đã tắt camera.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã dừng chương trình bởi người dùng")