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


def detect_signs(image):
    """Nhận diện biển báo giao thông"""
    results = model(image)
    detected_signs = []

    for r in results:
        img_plot = r.plot()
        for box in r.boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            if conf >= 0.5:
                class_code = id_to_label[cls]
                class_name = labels_map.get(class_code, "Unknown")
                area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                detected_signs.append((area, class_code, class_name, conf))

    # Sắp xếp theo diện tích (lớn nhất lên đầu)
    detected_signs.sort(reverse=True, key=lambda x: x[0])

    return img_plot, detected_signs


def create_results_window(detected_signs):
    """Tạo cửa sổ hiển thị kết quả nhận diện"""
    # Tạo hình ảnh trắng kích thước 400x600
    results_image = np.ones((600, 400, 3), dtype=np.uint8) * 255

    # Tiêu đề
    cv2.putText(results_image, "BIỂN BÁO PHÁT HIỆN ĐƯỢC", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Vẽ đường kẻ
    cv2.line(results_image, (20, 40), (380, 40), (0, 0, 0), 2)

    if not detected_signs:
        cv2.putText(results_image, "Không phát hiện biển báo nào", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        y_pos = 80
        for i, (_, code, name, conf) in enumerate(detected_signs):
            # Mã biển báo
            cv2.putText(results_image, f"{i + 1}. {code}", (30, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_pos += 30

            # Tên biển báo
            if len(name) > 30:
                # Chia tên dài thành nhiều dòng
                words = name.split()
                line = ""
                for word in words:
                    test_line = line + word + " "
                    if len(test_line) <= 30:
                        line = test_line
                    else:
                        cv2.putText(results_image, line, (40, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        y_pos += 25
                        line = word + " "
                if line:
                    cv2.putText(results_image, line, (40, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                cv2.putText(results_image, name, (40, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            y_pos += 25

            # Độ tin cậy
            cv2.putText(results_image, f"Độ tin cậy: {conf:.2f}", (40, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1)

            y_pos += 40

            # Giới hạn số lượng biển báo hiển thị
            if i >= 5:
                cv2.putText(results_image, f"... và {len(detected_signs) - 6} biển báo khác",
                            (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                break

    # Thêm hướng dẫn
    cv2.putText(results_image, "Nhấn 'q' để thoát", (110, 580),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 0), 2)

    return results_image


def main():
    # Mở camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera. Vui lòng kiểm tra thiết bị!")
        return

    print("Đã khởi động camera. Đang nhận diện biển báo...")
    print("Nhấn 'q' để thoát.")

    # Đặt vị trí cửa sổ
    cv2.namedWindow('Camera')
    cv2.moveWindow('Camera', 50, 50)

    cv2.namedWindow('Kết Quả Nhận Diện')
    cv2.moveWindow('Kết Quả Nhận Diện', 700, 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ camera. Thoát...")
                break

            # Thay đổi kích thước frame cho phù hợp với mô hình
            frame_resized = cv2.resize(frame, (640, 640))

            # Nhận diện biển báo
            result_image, detected_signs = detect_signs(frame_resized)

            # Tạo cửa sổ kết quả
            results_window = create_results_window(detected_signs)

            # Hiển thị kết quả
            cv2.imshow('Camera', result_image)
            cv2.imshow('Kết Quả Nhận Diện', results_window)

            # In kết quả ra terminal
            if detected_signs:
                print("\nBiển báo phát hiện được:")
                for _, code, name, conf in detected_signs:
                    print(f"- {code}: {name} ({conf:.2f})")

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