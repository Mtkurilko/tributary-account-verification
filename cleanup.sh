#!/bin/sh

find . -type d -name '__pycache__' -exec rm -rf {} +

find . -type f -name '*.json' -exec rm -f {} +

find . -type f -name '*.csv' -exec rm -f {} +
