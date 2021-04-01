#!/bin/sh
lock_file="requirements-lock.txt"
pip-compile setup.py --find-links=https://download.pytorch.org/whl/torch_stable.html --upgrade --generate-hashes --output-file=$lock_file