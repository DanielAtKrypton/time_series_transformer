#!/bin/bash
SCRIPT_PATH="$(dirname "${BASH_SOURCE[0]}")"
python -m venv .venv
. $SCRIPT_PATH/../.venv/bin/activate  # Use this to activate the virtual environment
python -m pip install --upgrade pip
pip install wheel
pip install pip-tools
$SCRIPT_PATH/pip-sync.sh
pip install -e .[dev,test,docs]