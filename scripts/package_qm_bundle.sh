#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/dist"
OUT_ZIP="${OUT_DIR}/qm_files_bundle.zip"

mkdir -p "${OUT_DIR}"

# Rebuild all QM files from TS before packaging.
for ts in "${ROOT_DIR}"/translate/*.ts; do
  qm="${ts%.ts}.qm"
  pyside6-lrelease "${ts}" -qm "${qm}" >/dev/null
done

(
  cd "${ROOT_DIR}/translate"
  zip -9 -q -j "${OUT_ZIP}" ./*.qm
)

echo "Created: ${OUT_ZIP}"
sha256sum "${OUT_ZIP}" || shasum -a 256 "${OUT_ZIP}"
