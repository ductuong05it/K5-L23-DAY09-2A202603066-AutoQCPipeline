"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line


def run(ds, cfg):
    issues = []
    c_fmt = cfg["label_format"]
    overflow_tol = c_fmt["overflow_tol"]
    dup_box_iou = c_fmt["dup_box_iou"]
    min_box_px = c_fmt["min_box_px"]
    max_box_frac = c_fmt["max_box_frac"]

    # Thu thập danh sách stem theo thứ tự ảnh
    seen = set()
    stems = []
    for p in ds.image_files():
        stems.append(p.stem)
        seen.add(p.stem)
    for p in ds.label_files():
        if p.stem not in seen:
            stems.append(p.stem)
            seen.add(p.stem)

    for stem in stems:
        lines = ds.read_label_lines(stem)
        if lines is None:
            continue

        valid_boxes = []  # Lưu (class_id, (cx, cy, w, h)) để kiểm tra lặp box

        for line_no, raw_line in lines:
            # 1. Kiểm tra định dạng 5 cột số
            try:
                cid, cx, cy, w, h = parse_yolo_line(raw_line)
            except ValueError:
                issues.append(Issue(
                    check="label_format",
                    severity=ERROR,
                    code="BAD_FORMAT",
                    image=stem,
                    line=line_no,
                    message=f"Dòng không đúng 5 cột số hợp lệ: {raw_line.strip()}"
                ))
                continue

            # 2. Kiểm tra ID lớp hợp lệ (0..len(classes)-1)
            if cid < 0 or cid >= len(ds.classes):
                issues.append(Issue(
                    check="label_format",
                    severity=ERROR,
                    code="BAD_CLASS_ID",
                    image=stem,
                    line=line_no,
                    message=f"ID lớp {cid} nằm ngoài danh sách lớp hợp lệ (0..{len(ds.classes) - 1})"
                ))
                continue

            # 3. Kiểm tra kích thước không âm hoặc bằng 0
            if w <= 0 or h <= 0:
                issues.append(Issue(
                    check="label_format",
                    severity=ERROR,
                    code="BOX_ZERO",
                    image=stem,
                    line=line_no,
                    message=f"Kích thước box không hợp lệ (w={w}, h={h})"
                ))
                continue

            # --- Kiểm tra hình học ---
            box = (cx, cy, w, h)

            # 4. Kiểm tra tràn mép ảnh
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            overflow = max(0.0, -x1, -y1, x2 - 1.0, y2 - 1.0)
            if overflow > overflow_tol:
                issues.append(Issue(
                    check="label_format",
                    severity=ERROR,
                    code="BOX_OUT_OF_IMAGE",
                    image=stem,
                    line=line_no,
                    message=f"Box tràn khỏi ảnh ({overflow:.4f} > {overflow_tol})",
                    extra={"overflow": overflow}
                ))

            # 5. Kiểm tra lặp box (trùng với một box trước đó cùng lớp, IoU cao)
            for prev_cid, prev_box in valid_boxes:
                if prev_cid == cid:
                    iou = iou_xywh(box, prev_box)
                    if iou > dup_box_iou:
                        issues.append(Issue(
                            check="label_format",
                            severity=ERROR,
                            code="DUP_BOX",
                            image=stem,
                            line=line_no,
                            message=f"Trùng lặp với box trước đó (IoU {iou:.3f} > {dup_box_iou})"
                        ))
                        break

            valid_boxes.append((cid, box))

            # 6. Cảnh báo box quá nhỏ (cạnh ngắn tính theo pixel < min_box_px)
            img_sz = ds.image_size(stem)
            if img_sz is not None:
                img_w, img_h = img_sz
                short_side_px = min(w * img_w, h * img_h)
                if short_side_px < min_box_px:
                    issues.append(Issue(
                        check="label_format",
                        severity=WARN,
                        code="BOX_TINY",
                        image=stem,
                        line=line_no,
                        message=f"Box quá nhỏ: cạnh ngắn {short_side_px:.1f}px < {min_box_px}px"
                    ))

            # 7. Cảnh báo box quá to (chiếm gần trọn ảnh)
            if w * h > max_box_frac:
                issues.append(Issue(
                    check="label_format",
                    severity=WARN,
                    code="BOX_HUGE",
                    image=stem,
                    line=line_no,
                    message=f"Box quá lớn: diện tích chiếm {w * h:.3f} > {max_box_frac}"
                ))

    return issues

