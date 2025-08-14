#!/bin/bash
echo "================================="
echo " Offline Oracle - Interactive Setup"
echo "================================="
echo ""
echo "This script will create a Python virtual environment and install all"
echo "necessary packages. You will be asked if you want to install the"
echo "GPU-accelerated versions for NVIDIA cards."
echo ""

python3 setup.py

echo ""
echo "================================="
echo " Setup script finished."
echo "================================="
read -p "Press Enter to continue..."