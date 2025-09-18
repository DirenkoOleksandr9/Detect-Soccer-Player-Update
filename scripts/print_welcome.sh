#!/usr/bin/env bash
set -e
python3 --version
pip --version
printf "\nReady. To run full pipeline (CPU only):\n"
printf "python3 main_pipeline.py 9.mp4 --player-id 1 --output-dir output\n\n"
