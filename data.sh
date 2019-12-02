#!/bin/bash

set -e

source venv/bin/activate

rm -r data/annotations
python3 src/preprocess_data.py
