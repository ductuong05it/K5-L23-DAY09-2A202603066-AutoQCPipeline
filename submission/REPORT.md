# Báo cáo Day 09 — Lê Đức Tường, 2A202603066

## 1. Kết luận trước
Pipeline kết luận **Release** sau 3 vòng QC và 1 lượt duyệt người mô phỏng. Tổng số ảnh quét đầu vào là 224 ảnh (`data/raw`), qua cửa nhận (intake) loại 31 ảnh (3 ảnh BAD_SIZE, 2 ảnh CORRUPT_IMAGE, 8 ảnh DUP_IMAGE_EXACT, 7 ảnh DUP_IMAGE_NEAR, 11 ảnh LEAK_TEST), chấp nhận 193 ảnh. Sau khi tự động sửa lỗi cú pháp và duyệt 80 ảnh nghi vấn ngữ nghĩa bằng người duyệt mô phỏng, pipeline không còn lỗi bắt buộc (0 ERROR), số ảnh còn cảnh báo chưa duyệt là 13/193 ảnh (tỉ lệ 6.74% < ngưỡng 10.0%), xuất thành công dataset sạch gồm 193 ảnh vào `submission/export/clean_v1/`.

## 2. Lỗi và phân việc
Lấy từ `02_error_list_v1` (gom theo mã lỗi). Cột "ai sửa": **máy** (autofix) hay **người** (hàng đợi); chỉ ghi những mã có mặt trong lô của bạn.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| BAD_CLASS_ID | 6 | Máy | ID lớp nằm ngoài phạm vi 0..3 (93, -1, 7, 4), máy xoá dòng an toàn (drop_line). |
| BAD_CLASS_NAME | 2 | Máy | Tên lớp trong data.yaml viết sai quy chuẩn ('Person', 'Bus'), máy chuẩn hóa lại đúng classes. |
| BAD_FORMAT | 3 | Máy | Dòng nhãn thiếu cột hoặc chứa nan/inf, không thể nạp vào model, máy xoá dòng (drop_line). |
| BAD_SIZE | 3 | Máy | Ảnh cạnh ngắn < 250px không đủ chuẩn huấn luyện, máy loại bỏ ảnh (remove_image). |
| BOX_OUT_OF_IMAGE | 8 | Máy | Box tràn viền toạ độ [0, 1] vượt ngưỡng, máy tự xén box về mép ảnh (clip_box). |
| BOX_ZERO | 4 | Máy | Box có chiều rộng hoặc chiều cao bằng 0 (vô nghĩa về hình học), máy xoá dòng (drop_line). |
| CORRUPT_IMAGE | 2 | Máy | File ảnh bị hỏng cấu trúc (UnidentifiedImageError, OSError), máy loại bỏ ảnh (remove_image). |
| DUP_BOX | 7 | Máy | Hai box cùng lớp đè khít lên nhau (IoU > 0.9), máy xoá box trùng lặp ở dòng sau (drop_line). |
| DUP_IMAGE_EXACT | 8 | Máy | Trùng từng byte mã hash sha1 với ảnh khác trong lô, máy xoá ảnh trùng (remove_image). |
| DUP_IMAGE_NEAR | 7 | Máy | Ảnh bị biến đổi nhẹ (phash_dist <= 16) trùng lặp nội dung, máy xoá ảnh trùng (remove_image). |
| LEAK_TEST | 11 | Máy | Ảnh trùng lặp với kho dữ liệu test/val độc lập, máy loại bỏ để chống rò rỉ dữ liệu (remove_image). |
| MISSING_LABEL_FILE | 6 | Người | Ảnh thiếu file nhãn, máy tuyệt đối không được tự bịa nhãn mà phải chuyển người gán nhãn mới. |
| BOX_HUGE | 3 | Người | Cảnh báo box chiếm > 95% ảnh, cần người xem có phải vật thể bao trùm hay vẽ nhầm toàn cảnh. |
| BOX_TINY | 6 | Người | Cảnh báo box có cạnh < 12px, cần người xem là vật thể ở xa hay là nhiễu hạt. |
| CLASS_RARE | 4 | Người | Cảnh báo số lượng box của một lớp quá ít trong dataset, cần con người cân đối dữ liệu. |
| EMPTY_LABEL | 11 | Người | File nhãn rỗng, cần người xác nhận là ảnh nền (không có vật thể) hay bị sót nhãn. |
| POSSIBLE_FALSE_POS | 28 | Người | Nhãn có box nhưng model soi không thấy vật thể, cần người xác nhận là box thừa hay model bỏ sót. |
| POSSIBLE_LOOSE_BOX | 11 | Người | Box vẽ lệch/lỏng so với vật thể (IoU < 0.5), cần người chỉnh lại khung bao cho khít. |
| POSSIBLE_MISSING | 32 | Người | Model thấy vật rõ nét nhưng nhãn không có box, cần người bổ sung box nếu đúng là vật thể thật. |
| POSSIBLE_WRONG_CLASS | 13 | Người | Nghi ngờ nhầm lẫn giữa các lớp xe/người, chỉ có con người mới phân biệt chính xác ngữ nghĩa. |

Vì sao `05_review_queue.csv` xếp ảnh đầu hàng như vậy?
Hàng đợi được xếp theo điểm ưu tiên kết hợp từ trọng số lỗi (`weights` trong config):
- Ảnh `img_0020` đứng đầu với điểm ưu tiên 11.524 vì dính đồng thời nhiều lỗi phức tạp: vừa bị máy xoá dòng sai (`FIXED_DROP_LINE`), vừa có cảnh báo thiếu nhãn (`POSSIBLE_MISSING`) và nghi ngờ sai lớp (`POSSIBLE_WRONG_CLASS`).
- Tiếp theo là 6 ảnh dính lỗi `NEEDS_LABEL` (`img_0031`, `img_0053`, `img_0126`, `img_0135`, `img_0170`, `img_0219`) được gán trọng số cao nhất (weight = 10.0), vì nếu người không gán nhãn cho các ảnh này thì Gate sẽ lập tức Rework toàn bộ pipeline.
- Các ảnh dính lỗi ngữ nghĩa nghiêm trọng (`POSSIBLE_WRONG_CLASS` weight = 3, `POSSIBLE_MISSING` weight = 2) được ưu tiên xếp trước các cảnh báo hình học nhẹ (`BOX_TINY` weight = 0.3).
Với ngân sách 80 ảnh, người duyệt đã xem được 80/84 lỗi ngữ nghĩa (đạt tỉ lệ bao phủ 95.2% trên toàn bộ các lỗi ngữ nghĩa của lô, trong đó bắt trọn 100% các lỗi `POSSIBLE_WRONG_CLASS` (13/13), `POSSIBLE_MISSING` (32/32), `POSSIBLE_LOOSE_BOX` (11/11), và 24/28 lỗi `POSSIBLE_FALSE_POS`).

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 16 | DUP_IMAGE_NEAR: (1.00 / 0.50) → (1.00 / 0.83) | Tăng phash_dist từ 2 lên 16 giúp Recall tăng từ 50% lên 83.3% mà Precision vẫn đạt 1.00 (không xóa nhầm ảnh tốt trên tập calib). Ngưỡng 20 bị loại vì Recall sụt giảm (do băm quá lỏng làm loạn ghép cặp). |
| `second_opinion.conf` | 0.25 | 0.80 | WRONG_CLASS: (0.56 / 0.83) → (1.00 / 0.83); MISSING: (0.50 / 1.00) → (0.67 / 0.75) | Ở conf=0.25, mô hình quá nhạy khiến báo động giả cao (Precision chỉ 0.50 - 0.56), làm hàng đợi tràn ngân sách (145 ảnh > 80) dẫn đến Gate bị Hold (29.3%). Tăng conf lên 0.80 chấp nhận Recall giảm nhẹ nhưng Precision tăng vọt (WRONG_CLASS đạt 1.00, MISSING đạt 0.67), lọc sạch cảnh báo rác để hàng đợi co về 98 ảnh, giúp người duyệt xử lý gọn trong ngân sách 80 ảnh và đưa Gate về Release (6.7%). |

Một chỗ calibration có thể đánh lừa bạn:
Tập dữ liệu `calib` chỉ có 39 ảnh với các lỗi nhân tạo tương đối rõ ràng. Trên tập calib, `phash_dist = 16` đạt Precision 1.00 hoàn hảo. Nhưng trên dữ liệu thực tế (`raw`), `phash_dist = 16` đã loại tới 7 ảnh `DUP_IMAGE_NEAR` và tổng cộng 31 ảnh ở intake. Khi mở kiểm tra trực quan các cặp ảnh bị gán nhãn trùng như `img_0011` với `img_0008` (phash 16) hay `img_0177` với `img_0113` (phash 16), ta thấy rằng hai ảnh có thể chỉ cùng một bối cảnh/nền đường phố nhưng các phương tiện giao thông di chuyển ở vị trí khác nhau. Ngưỡng 16 trên thực tế có nguy cơ xóa nhầm những ảnh có góc chụp tương đồng nhưng nhãn khác nhau.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
(a) Chứng minh gate chặn đúng: Khi chạy lệnh thử nghiệm giả lập trường hợp không có người duyệt:
`python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`
Kết quả: Pipeline in `kết luận: REWORK — Còn 6 lỗi bắt buộc (ERROR) cần sửa lại.`, `Bước 5/5 KHÔNG xuất: còn lỗi bắt buộc (Rework) — trả về bước 3`, mã thoát trả về đúng 20 (`EXIT_CODE=20`) và thư mục `work/exp_nb` hoàn toàn không được tạo ra. Điều này chứng minh Gate đã chặn đứng triệt để dữ liệu lỗi khi thiếu sự can thiệp của con người.

(b) Hai điểm mù mà pipeline không bắt được:
1. Box vẽ lệch/lỏng nhẹ hoặc bám không sát vật thể nhưng IoU vẫn >= 0.50: Ở ngưỡng `iou_loose = 0.50`, nếu nhãn vẽ thừa viền xe hoặc lệch một phần thân người nhưng diện tích giao nhau vẫn đạt IoU trên 0.50, pipeline coi như khớp hoàn toàn và bỏ qua, không đưa ra cảnh báo.
2. Vật thể bị che khuất một phần hoặc kích thước nhỏ mờ ở hậu cảnh mà cả người gán nhãn lẫn mô hình YOLO26L đều bỏ sót: Khi cả con người và mô hình lớn (checker) cùng có điểm mù chung (không detect được), pipeline hoàn toàn không thể phát hiện lỗi `POSSIBLE_MISSING` này.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa chạy do tập trung hoàn thiện phần lõi pipeline và kiểm tra chất lượng dữ liệu.

## 6. Quyết định và ghi chú AI
(a) Quyết định cuối: **Release**. Lô dữ liệu đạt chuẩn xuất xưởng với 0 lỗi bắt buộc, 193 ảnh sạch được xuất sang `submission/export/clean_v1/`. Tỉ lệ cảnh báo tồn đọng là 6.74% (13/193 ảnh), nằm an toàn dưới ngưỡng chặn 10.0%.
(b) Ghi chú dùng AI:
- Prompt chính: "Phân tích docstring và viết 4 module label_format.py, gate.py, autofix.py, second_opinion.py theo chuẩn TDD đảm bảo pass toàn bộ test suite và không can thiệp trực tiếp vào thư mục data/raw."
- Cách kiểm tra code AI: Chạy toàn bộ 79 unit test bằng pytest (`python -m pytest -q`), chạy script `check_submission.py --no-train` để rà soát mã bảo vệ sha1 và kiểm tra tính toàn vẹn của logic cửa chặn.
- Một lần AI sai: Ban đầu khi viết hàm `second_opinion.py`, AI đã dùng `ds.valid_boxes()` làm mất thông tin số dòng gốc (`line_no`), dẫn đến lỗi `POSSIBLE_FALSE_POS` không ghi được số dòng đúng khi gặp file có dòng trống ở giữa; sau đó đã được sửa lại dùng `ds.read_label_lines()` để giữ nguyên số dòng gốc 1-indexed.
