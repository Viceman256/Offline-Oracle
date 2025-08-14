#!/bin/bash
echo "================================="
echo " Offline Oracle - Knowledge Base Builder"
echo "================================="
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Starting the build process..."
echo "This may take a very long time depending on the size of your ZIM files."
echo "You can stop this process at any time (Ctrl+C) and it will resume later."
echo ""
python3 build.py

echo ""
echo "================================="
echo " Build process finished."
echo "================================="
read -p "Press Enter to continue..."