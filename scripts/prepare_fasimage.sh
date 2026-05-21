#!/usr/bin/env bash
set -euo pipefail

# Root paths (edit as needed)
SRC_ROOT="/share/rongss/spoofing/DGFAS_Demo"
PROTO_ROOT="${SRC_ROOT}/dg_protocol/3to1_v1"
OUT_ROOT="/share/rongss/test_time/IAPL/Datasets/FasImage"

# Datasets to build (edit this list)
DATASETS=(
  "OULU_NPU"
  "MSU_MFSD"
  "ReplayAttack"
  "CASIA_FASD"
)

# Which splits to build (edit this list)
SPLITS=("test" "train")

# Build dataset
for dataset in "${DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    out_real="${OUT_ROOT}/${split}/${dataset}/0_real"
    out_fake="${OUT_ROOT}/${split}/${dataset}/1_fake"
    mkdir -p "${out_real}" "${out_fake}"

        json_path="${PROTO_ROOT}/${dataset}_${split}.json"
        if [[ ! -f "${json_path}" ]]; then
            echo "[WARN] Missing protocol: ${json_path} (skip)"
            continue
        fi

        # Build split directly from protocol JSON.
        python - <<'PY' "${json_path}" "${SRC_ROOT}" "${out_real}" "${out_fake}" "${dataset}" "${split}"
import json
import os
import sys

json_path, src_root, out_real, out_fake, dataset, split = sys.argv[1:]

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

copied = 0
missing = 0
skipped = 0

for item in data:
    images = item.get("images", [])
    if not images:
        skipped += 1
        continue
    rel_path = images[0]
    label = item.get("binary_label", None)
    if label not in (0, 1):
        skipped += 1
        continue

    # Rename: ".../6_2_51_2/crop/016.png" -> "6_2_51_2_016.png"
    rel_path_norm = rel_path.replace("\\", "/")
    parts = rel_path_norm.split("/")
    if len(parts) < 3:
        skipped += 1
        continue
    parent = parts[-3]
    filename = parts[-1]
    new_name = f"{parent}_{filename}"

    src_path = os.path.join(src_root, rel_path_norm)
    dst_dir = out_real if label == 0 else out_fake
    dst_path = os.path.join(dst_dir, new_name)

    if not os.path.exists(src_path):
        print(f"[WARN] Missing file: {src_path}")
        missing += 1
        continue

    if not os.path.exists(dst_path):
        with open(src_path, "rb") as r, open(dst_path, "wb") as w:
            w.write(r.read())
        copied += 1

print(
    f"[INFO] {dataset} {split} copied={copied} "
    f"missing={missing} skipped={skipped}"
)
PY
  done
done

echo "Done."
