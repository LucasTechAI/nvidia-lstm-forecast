#!/bin/bash
# Format all Python files in the current directory and its subdirectories using Black, ignoring venv folders

echo "======================================"
echo "Formatting all Python files with Black (excluding venv)"
echo "======================================"

find . -type d -name "venv" -prune -o -type f -name "*.py" -exec black {} +

echo "--------------------------------------"
echo "Formatting complete!"
echo "--------------------------------------"