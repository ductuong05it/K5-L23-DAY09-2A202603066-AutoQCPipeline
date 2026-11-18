"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
from ..common import ERROR, WARN


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    valid_stems = set(image_stems)
    n_images = len(image_stems)

    # 1. Đếm lỗi bắt buộc (ERROR)
    # Lỗi bắt buộc không bao giờ được bỏ qua (kể cả ảnh đã reviewed)
    error_issues = [i for i in issues if i.severity == ERROR]
    n_errors = len(error_issues)

    # 2. Đếm số ảnh có cảnh báo (WARN) chưa được reviewed
    # Chỉ đếm issue có image nằm trong image_stems (bỏ "<dataset>", "data.yaml")
    # và bỏ qua ảnh nằm trong reviewed
    warn_images = {
        i.image
        for i in issues
        if i.severity == WARN and i.image in valid_stems and i.image not in reviewed
    }
    n_warn_images = len(warn_images)

    # 3. Tính tỉ lệ ảnh cảnh báo
    warn_ratio = round(n_warn_images / n_images, 4) if n_images > 0 else 0.0

    # 4. Quyết định: Rework -> Hold -> Release
    reasons = []
    if n_errors > 0:
        decision = "Rework"
        reasons.append(f"Còn {n_errors} lỗi bắt buộc (ERROR) cần sửa lại.")
    elif warn_ratio >= cfg["gate"]["warn_ratio_hold"]:
        decision = "Hold"
        reasons.append(
            f"Tỉ lệ ảnh còn cảnh báo ({warn_ratio:.1%}) >= ngưỡng cho phép "
            f"({cfg['gate']['warn_ratio_hold']:.1%}) ({n_warn_images}/{n_images} ảnh)."
        )
    else:
        decision = "Release"
        if n_warn_images > 0:
            reasons.append(
                f"Đạt chuẩn xuất xưởng: không còn lỗi bắt buộc, tỉ lệ cảnh báo "
                f"({warn_ratio:.1%}) dưới ngưỡng ({cfg['gate']['warn_ratio_hold']:.1%})."
            )
        else:
            reasons.append("Đạt chuẩn xuất xưởng: dữ liệu sạch, không còn lỗi hay cảnh báo tồn đọng.")

    return {
        "decision": decision,
        "n_errors": n_errors,
        "n_warn_images": n_warn_images,
        "n_images": n_images,
        "warn_ratio": warn_ratio,
        "reasons": reasons,
    }

