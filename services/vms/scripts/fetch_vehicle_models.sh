#!/usr/bin/env bash
# Extract the NVIDIA TAO vehicle-attribute classifiers (VehicleMakeNet /
# VehicleTypeNet) from the DeepStream image into vms/models/. These ONNX models
# are gitignored (binary, ship with DeepStream), so run this once after cloning
# on a box that has the DeepStream image pulled.
#
#   bash scripts/fetch_vehicle_models.sh
#
# Requires: docker, and the image nvcr.io/nvidia/deepstream:7.1-triton-multiarch
set -euo pipefail

IMAGE="${DS_IMAGE:-nvcr.io/nvidia/deepstream:7.1-triton-multiarch}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/models"
DS="/opt/nvidia/deepstream/deepstream-7.1/samples/models"

cid=$(docker create "$IMAGE")
trap 'docker rm "$cid" >/dev/null 2>&1 || true' EXIT

mkdir -p "$DEST/vehiclemakenet" "$DEST/vehicletypenet"
docker cp "$cid:$DS/Secondary_VehicleMake/resnet18_vehiclemakenet_pruned.onnx" "$DEST/vehiclemakenet/"
docker cp "$cid:$DS/Secondary_VehicleMake/labels.txt"                          "$DEST/vehiclemakenet/"
docker cp "$cid:$DS/Secondary_VehicleTypes/resnet18_vehicletypenet_pruned.onnx" "$DEST/vehicletypenet/"
docker cp "$cid:$DS/Secondary_VehicleTypes/labels.txt"                          "$DEST/vehicletypenet/"

echo "Installed vehicle attribute models into $DEST"
