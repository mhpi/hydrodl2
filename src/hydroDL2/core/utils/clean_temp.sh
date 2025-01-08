#!/bin/sh

# Recursively remove all cache files (.pyc, __pycache__/) in the working dir.
# Usage: sh path/to/clean_temp.sh

find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf
