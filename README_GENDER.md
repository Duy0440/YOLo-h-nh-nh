# Hướng dẫn tạo dataset "Nam/Nữ" cho YOLO

Dự án hiện tại đã có:
- `data_gender.yaml`: config dataset (2 class: male/female)
- `train_gender.py`: script dùng để train YOLO trên dataset đó

## 1) Chuẩn bị dữ liệu

### Thư mục cần tạo
Tạo cấu trúc sau (bên trong `input/images/`):

```
input/images/gender/
  train/
  val/
```

Trong `train/` và `val/`, bạn đặt các ảnh (jpg/png) dùng để train/validate.

### Gán nhãn (label) theo định dạng YOLO
Với mỗi ảnh `xxx.jpg`, bạn cần tạo file text cùng tên `xxx.txt` (trong cùng thư mục) với nội dung:

```
<class> <x_center> <y_center> <width> <height>
```

- `class`: `0` = male, `1` = female
- `x_center`, `y_center`, `width`, `height` phải là **giá trị chuẩn hóa (0..1)**
  - `x_center = (x_min + x_max) / 2 / image_width`
  - `width = (x_max - x_min) / image_width`
  - tương tự với y

Ví dụ 1 người nam ở giữa ảnh:
```
0 0.5 0.5 0.3 0.4
```

> Lưu ý: Nếu ảnh có nhiều người, hãy thêm nhiều dòng (mỗi người 1 dòng).

---

## 2) Train model
Sau khi dataset đã sẵn sàng (có cả ảnh và file `.txt` nhãn), chạy:

```powershell
python train_gender.py --data data_gender.yaml --epochs 20 --model yolov8n.pt
```

Kết quả sẽ được lưu ở:
```
runs/train/gender/weights/best.pt
```

---

## 3) Sử dụng model để đếm nam/nữ
Sau khi train xong, dùng model `best.pt` để chạy code đã có:

```powershell
python main.py --input input/images/gender/test_image.png --output output/results/out.png --model runs/train/gender/weights/best.pt --conf 0.5
```

Output sẽ in ra terminal dạng:
```
Counts: total=3, male=2, female=1
```

---

## 4) Gợi ý công cụ gán nhãn (annotation)
- **LabelImg** (GUI): https://github.com/tzutalin/labelImg (có thể xuất ra YOLO format)
- **Roboflow**: upload ảnh, annotate, export YOLO


---

Nếu bạn có ảnh mẫu, gửi mình 1-2 ảnh để mình giúp bạn tạo label mẫu và test nhanh.
