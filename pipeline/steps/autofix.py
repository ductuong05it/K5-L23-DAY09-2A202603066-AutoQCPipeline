"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict
from pathlib import Path

import yaml

from ..common import ERROR, parse_yolo_line


def fix(ds, issues, cfg):
    fixes = []
    # Chỉ xử lý các lỗi có severity ERROR (không động tới WARN)
    err_issues = [i for i in issues if i.severity == ERROR]
    if not err_issues:
        return fixes

    removed_images = set()

    # 1. Nhóm xóa ảnh: DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE
    for iss in err_issues:
        code = iss.code
        if code.startswith("DUP_IMAGE_") or code in ("LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"):
            stem = iss.image
            if stem not in removed_images:
                removed_images.add(stem)
                # Xóa ảnh trong images_dir
                if ds.images_dir.exists():
                    for p in ds.images_dir.glob(f"{stem}.*"):
                        p.unlink(missing_ok=True)
                # Xóa file nhãn trong labels_dir
                lp = ds.label_path(stem)
                if lp.exists():
                    lp.unlink(missing_ok=True)
                fixes.append({
                    "image": stem,
                    "line": "",
                    "action": "remove_image",
                    "before": stem,
                    "after": "đã xoá"
                })

    # 2. Sửa tên lớp trong data.yaml: BAD_CLASS_NAME
    for iss in err_issues:
        if iss.code == "BAD_CLASS_NAME":
            before_val = ""
            data = {}
            if ds.yaml_path.exists():
                data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
                before_val = str(data.get("names", ""))
            data["names"] = list(cfg["classes"])
            ds.yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            fixes.append({
                "image": "data.yaml",
                "line": "",
                "action": "rename_classes",
                "before": before_val,
                "after": str(cfg["classes"])
            })

    # 3. Ảnh thiếu nhãn: MISSING_LABEL_FILE (không tạo file, chuyển người gán)
    for iss in err_issues:
        if iss.code == "MISSING_LABEL_FILE":
            stem = iss.image
            if stem not in removed_images:
                fixes.append({
                    "image": stem,
                    "line": "",
                    "action": "needs_label",
                    "before": "(không có file)",
                    "after": "chuyển người gán"
                })

    # 4. Sửa từng dòng trong file nhãn (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX, BOX_OUT_OF_IMAGE)
    # Gom lỗi theo file nhãn và theo số dòng
    file_issues = defaultdict(lambda: defaultdict(list))
    for iss in err_issues:
        stem = iss.image
        if stem in removed_images or stem in ("data.yaml", "<dataset>"):
            continue
        if iss.line is not None:
            file_issues[stem][iss.line].append(iss)

    for stem, line_issues in file_issues.items():
        lp = ds.label_path(stem)
        if not lp.exists():
            continue

        raw_content = lp.read_text(encoding="utf-8", errors="replace")
        original_lines = raw_content.splitlines()

        new_lines = []
        for idx in range(1, len(original_lines) + 1):
            line_str = original_lines[idx - 1]
            if idx in line_issues:
                issues_on_line = line_issues[idx]
                codes = {i.code for i in issues_on_line}

                # Ưu tiên xóa dòng nếu dính các lỗi không thể cứu:
                # BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX
                drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
                if codes & drop_codes:
                    fixes.append({
                        "image": stem,
                        "line": idx,
                        "action": "drop_line",
                        "before": line_str,
                        "after": ""
                    })
                    continue

                # Nếu là BOX_OUT_OF_IMAGE: thử clip box về [0, 1]
                if "BOX_OUT_OF_IMAGE" in codes:
                    try:
                        cid, cx, cy, w, h = parse_yolo_line(line_str)
                        x1 = max(0.0, cx - w / 2.0)
                        y1 = max(0.0, cy - h / 2.0)
                        x2 = min(1.0, cx + w / 2.0)
                        y2 = min(1.0, cy + h / 2.0)
                        new_w = x2 - x1
                        new_h = y2 - y1
                        if new_w <= 0 or new_h <= 0:
                            # Box nằm hoàn toàn ngoài ảnh -> drop_line
                            fixes.append({
                                "image": stem,
                                "line": idx,
                                "action": "drop_line",
                                "before": line_str,
                                "after": ""
                            })
                            continue
                        new_cx = (x1 + x2) / 2.0
                        new_cy = (y1 + y2) / 2.0
                        clipped_line = f"{cid} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"
                        fixes.append({
                            "image": stem,
                            "line": idx,
                            "action": "clip_box",
                            "before": line_str,
                            "after": clipped_line
                        })
                        new_lines.append(clipped_line)
                    except ValueError:
                        fixes.append({
                            "image": stem,
                            "line": idx,
                            "action": "drop_line",
                            "before": line_str,
                            "after": ""
                        })
                        continue
            else:
                new_lines.append(line_str)

        # Ghi lại file nhãn đã sửa
        if new_lines:
            lp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        else:
            lp.write_text("", encoding="utf-8")

    return fixes

