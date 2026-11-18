"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    c_so = cfg["second_opinion"]
    conf_thresh = c_so["conf"]
    min_area_px = c_so["min_area_px"]
    fp_conf = c_so["fp_conf"]
    iou_missing = c_so["iou_missing"]
    iou_match = c_so["iou_match"]
    iou_loose = c_so["iou_loose"]

    issues = []

    for img_p in ds.image_files():
        stem = img_p.stem
        # Bỏ qua ảnh không có trong preds hoặc không có file nhãn
        if stem not in preds or not ds.label_path(stem).exists():
            continue

        raw_lines = ds.read_label_lines(stem)
        valid_labels = []
        if raw_lines is not None:
            for line_no, text in raw_lines:
                try:
                    cid, cx, cy, w, h = parse_yolo_line(text)
                except ValueError:
                    continue
                if 0 <= cid < len(ds.classes):
                    valid_labels.append((line_no, cid, (cx, cy, w, h)))

        img_sz = ds.image_size(stem)
        img_w, img_h = img_sz if img_sz else (640, 480)

        # 1. P = dự đoán có conf >= conf VÀ diện tích px >= min_area_px
        pred_list = preds[stem] or []
        p_filtered = []
        for r in pred_list:
            p_cid, p_conf, p_cx, p_cy, p_w, p_h = r
            area_px = p_w * img_w * p_h * img_h
            if p_conf >= conf_thresh and area_px >= min_area_px:
                p_filtered.append(r)

        # 2. Duyệt từng dự đoán trong P
        for r in p_filtered:
            p_cid, p_conf, p_cx, p_cy, p_w, p_h = r
            p_box = (p_cx, p_cy, p_w, p_h)

            # Lấy cands: nhãn mọi lớp có IoU >= iou_missing
            cands = []
            for l_item in valid_labels:
                l_no, l_cid, l_box = l_item
                iou = iou_xywh(p_box, l_box)
                if iou >= iou_missing:
                    cands.append((l_no, l_cid, l_box, iou))

            if not cands:
                # Model thấy vật mà nhãn không có box
                issues.append(Issue(
                    check="second_opinion",
                    severity=WARN,
                    code="POSSIBLE_MISSING",
                    image=stem,
                    line=None,
                    message=f"Model thấy {ds.classes[p_cid] if 0 <= p_cid < len(ds.classes) else p_cid} (conf={p_conf:.2f}) nhưng nhãn không có box",
                    extra={"conf": round(p_conf, 4), "class": p_cid}
                ))
            else:
                same_class_cands = [c for c in cands if c[1] == p_cid]
                if same_class_cands:
                    best_same_iou = max(c[3] for c in same_class_cands)
                    if best_same_iou < iou_loose:
                        # Đúng vật, box lỏng/lệch
                        issues.append(Issue(
                            check="second_opinion",
                            severity=WARN,
                            code="POSSIBLE_LOOSE_BOX",
                            image=stem,
                            line=None,
                            message=f"Box lỏng/lệch (IoU={best_same_iou:.2f} < {iou_loose})",
                            extra={"conf": round(p_conf, 4), "iou": round(best_same_iou, 4)}
                        ))
                    else:
                        # Nhãn đúng lớp đã phủ vật này -> không cờ
                        pass
                else:
                    # Không có cands cùng lớp
                    best_diff_iou = max(c[3] for c in cands)
                    if best_diff_iou >= iou_match:
                        issues.append(Issue(
                            check="second_opinion",
                            severity=WARN,
                            code="POSSIBLE_WRONG_CLASS",
                            image=stem,
                            line=None,
                            message=f"Nghi nhầm lớp: model đoán {ds.classes[p_cid] if 0 <= p_cid < len(ds.classes) else p_cid} (conf={p_conf:.2f}), nhãn khác lớp (IoU={best_diff_iou:.2f})",
                            extra={"conf": round(p_conf, 4), "iou": round(best_diff_iou, 4)}
                        ))
                    else:
                        # [iou_missing, iou_match): mơ hồ -> không cờ
                        pass

        # 3. Kiểm tra POSSIBLE_FALSE_POS cho từng dòng nhãn hợp lệ
        # Nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
        fp_preds = [r for r in pred_list if r[1] >= fp_conf]
        for l_no, l_cid, l_box in valid_labels:
            all_below = True
            for r in fp_preds:
                p_box = (r[2], r[3], r[4], r[5])
                if iou_xywh(l_box, p_box) >= iou_missing:
                    all_below = False
                    break
            if all_below:
                issues.append(Issue(
                    check="second_opinion",
                    severity=WARN,
                    code="POSSIBLE_FALSE_POS",
                    image=stem,
                    line=l_no,
                    message=f"Nhãn tại dòng {l_no} không khớp với bất kỳ dự đoán nào của model (nghi box thừa)",
                    extra={}
                ))

    return issues

