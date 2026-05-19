#!/usr/bin/env bash
set -euo pipefail

# Root paths (edit as needed)
SRC_ROOT="/share/rongss/spoofing/DGFAS_Demo"
PROTO_ROOT="${SRC_ROOT}/dg_protocol/1to11_v1"
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

# Train list files per dataset (edit if paths differ)
declare -A TRAIN_LISTS=(
  ["OULU_NPU"]="${SRC_ROOT}/OULU_NPU/train.txt"
  ["MSU_MFSD"]="${SRC_ROOT}/MSU_MFSD/trainmsu.txt"
  ["ReplayAttack"]="${SRC_ROOT}/ReplayAttack/train.txt"
  ["CASIA_FASD"]="${SRC_ROOT}/CASIA_FASD/protocol/train.txt"
)

# Build dataset
for dataset in "${DATASETS[@]}"; do
  test_parents_file="$(mktemp)"
  test_json_path="${PROTO_ROOT}/${dataset}_test.json"
  if [[ -f "${test_json_path}" ]]; then
    python - <<'PY' "${test_json_path}" "${test_parents_file}"
import json
import sys

json_path, out_path = sys.argv[1:]

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

parents = set()
for item in data:
    images = item.get("images", [])
    if not images:
        continue
    rel_path = images[0].replace("\\", "/")
    parts = rel_path.split("/")
    if len(parts) < 3:
        continue
    parents.add(parts[-3])

with open(out_path, "w", encoding="utf-8") as f:
    for p in sorted(parents):
        f.write(p + "\n")
PY
  fi

  for split in "${SPLITS[@]}"; do
    out_real="${OUT_ROOT}/${split}/${dataset}/0_real"
    out_fake="${OUT_ROOT}/${split}/${dataset}/1_fake"
    mkdir -p "${out_real}" "${out_fake}"

    if [[ "${split}" == "test" ]]; then
      json_path="${PROTO_ROOT}/${dataset}_${split}.json"
      if [[ ! -f "${json_path}" ]]; then
        echo "[WARN] Missing protocol: ${json_path} (skip)"
        continue
      fi

      # Extract image paths + labels from JSON, then copy with renamed files.
      python - <<'PY' "${json_path}" "${SRC_ROOT}" "${out_real}" "${out_fake}"
import json
import os
import sys

json_path, src_root, out_real, out_fake = sys.argv[1:]

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    images = item.get("images", [])
    if not images:
        continue
    rel_path = images[0]
    label = item.get("binary_label", None)
    if label not in (0, 1):
        continue

    # Rename: ".../6_2_51_2/crop/016.png" -> "6_2_51_2_016.png"
    rel_path_norm = rel_path.replace("\\", "/")
    parts = rel_path_norm.split("/")
    if len(parts) < 3:
        continue
    parent = parts[-3]
    filename = parts[-1]
    new_name = f"{parent}_{filename}"

    src_path = os.path.join(src_root, rel_path)
    dst_dir = out_real if label == 0 else out_fake
    dst_path = os.path.join(dst_dir, new_name)

    if not os.path.exists(src_path):
        print(f"[WARN] Missing file: {src_path}")
        continue

    if not os.path.exists(dst_path):
        with open(src_path, "rb") as r, open(dst_path, "wb") as w:
            w.write(r.read())
PY
    fi

    if [[ "${split}" == "train" ]]; then
      train_list="${TRAIN_LISTS[${dataset}]-}"
      if [[ -z "${train_list}" || ! -f "${train_list}" ]]; then
      echo "[WARN] Missing train list for ${dataset}: ${train_list} (skip)"
      continue
      fi

      # From train list, pick one image per folder, excluding any folder used in test.
      python - <<'PY' "${train_list}" "${SRC_ROOT}" "${out_real}" "${out_fake}" "${test_parents_file}" "${dataset}"
import os
import sys

train_list, src_root, out_real, out_fake, test_parents_file, dataset = sys.argv[1:]

test_parents = set()
if os.path.exists(test_parents_file):
    with open(test_parents_file, "r", encoding="utf-8") as f:
        test_parents = {line.strip() for line in f if line.strip()}

def normalize_path(path: str, dataset_name: str) -> str:
    path = path.strip().replace("\\", "/")
    marker = f"/{dataset_name}/"
    if marker in path:
        idx = path.find(marker)
        return path[idx + 1:]
    if path.startswith(f"{dataset_name}/"):
        return path
    marker2 = f"{dataset_name}/"
    if marker2 in path:
        idx = path.find(marker2)
        return path[idx:]
    return ""

selected_parents = set()
missing = 0
skipped_test = 0
skipped_dup = 0
selected = 0

with open(train_list, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        label_str = parts[-1]
        rel = " ".join(parts[:-1])
        try:
            label = int(label_str)
        except ValueError:
            continue
        if label not in (0, 1):
            continue

        rel_path = normalize_path(rel, dataset)
        if not rel_path:
            continue

        rel_path = rel_path.replace("\\", "/")
        path_parts = rel_path.split("/")
        if len(path_parts) < 3:
            continue
        parent = path_parts[-3]
        if parent in test_parents:
            skipped_test += 1
            continue
        if parent in selected_parents:
            skipped_dup += 1
            continue

        src_path = os.path.join(src_root, rel_path)
        if not os.path.exists(src_path):
            print(f"[WARN] Missing file: {src_path}")
            missing += 1
            continue

        filename = path_parts[-1]
        new_name = f"{parent}_{filename}"
        dst_dir = out_real if label == 0 else out_fake
        dst_path = os.path.join(dst_dir, new_name)

        if not os.path.exists(dst_path):
            with open(src_path, "rb") as r, open(dst_path, "wb") as w:
                w.write(r.read())
            selected_parents.add(parent)
            selected += 1

print(
    f"[INFO] {dataset} train selected={selected} missing={missing} "
    f"skipped_test={skipped_test} skipped_dup={skipped_dup}"
)
PY
    fi
  done
    rm -f "${test_parents_file}"
 done

echo "Done."
