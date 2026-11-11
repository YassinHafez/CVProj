#!/usr/bin/env bash
# Download the Angelina Braille Images Dataset v1.1 (Ovodov, ICCV Workshops 2021) and build data/index.json.
# Run from the repository root:  bash scripts/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ ! -d data/angelina ]; then
  git clone --depth 1 https://github.com/IlyaOvodov/AngelinaDataset.git data/angelina
fi
python3 scripts/prepare_data.py data/angelina data/index.json
