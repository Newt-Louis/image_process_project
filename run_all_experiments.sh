#!/usr/bin/env bash
# Chạy toàn bộ kịch bản thực nghiệm A -> D (+ E, F nếu bật) rồi xuất báo cáo.
#
#   ./run_all_experiments.sh                 # toàn bộ 19.200 ảnh, tag=main
#   ./run_all_experiments.sh 500 thu         # 500 ảnh (lấy mẫu ngẫu nhiên), tag=thu
#   WITH_TN=1 ./run_all_experiments.sh       # kèm kịch bản E + F ([->TN] tốt nghiệp)
#
set -euo pipefail
cd "$(dirname "$0")"

LIMIT="${1:-}"
TAG="${2:-main}"
LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

log "A · Định lượng toàn tập  (tag=$TAG ${LIMIT_ARG:-toàn bộ})"
python -m experiments.run_eval --tag "$TAG" $LIMIT_ARG

log "B · Lát cắt theo độ khó & mật độ"
python -m experiments.run_slices --tag "$TAG"

log "C · Ảnh định tính (6 tiêu chí có chủ đích)"
python -m experiments.run_qualitative --tag "$TAG" --per-criterion 1

log "D · Phân tích lỗi"
python -m experiments.run_error_analysis --tag "$TAG"

if [ "${WITH_TN:-0}" = "1" ]; then
  log "E · [→TN] Robustness dưới nhiễu"
  python -m experiments.run_robustness --tag "$TAG" --limit 500

  log "F · [→TN] CPU benchmark + quantize INT8 (chỉ model đã chốt)"
  python -m experiments.run_cpu_bench --tag "$TAG" --model yolov11 --limit 200 --onnx || \
    echo "  (bỏ qua phần ONNX — cần: pip install onnx onnxruntime onnxslim)"
fi

log "Tổng hợp bảng + biểu đồ + báo cáo markdown"
python -m experiments.make_report --tag "$TAG"

log "XONG. Mở results/$TAG/BAO_CAO_THUC_NGHIEM.md"
