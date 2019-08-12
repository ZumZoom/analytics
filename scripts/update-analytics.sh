#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
cd ../dist/uniswap/data
rm providers/* roi/* volume/*
cd ../../../uniswap
python3 analyse.py
cd ../dist
git add -A
git commit -m "update data"
git push
