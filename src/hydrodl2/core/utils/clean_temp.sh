#!/bin/sh

# Recursively remove all cache files (.pyc, __pycache__/) in the working dir.
# Usage: sh path/to/clean_temp.sh

find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf
find . | grep -E "(.ruff_cache$)" | xargs rm -rf
find . | grep -E "(.pytest_cache$)" | xargs rm -rf
