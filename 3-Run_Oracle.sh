#!/bin/bash
echo "================================="
echo " Offline Oracle - RAG Server"
echo "================================="
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Starting the RAG server..."
echo "Check 'config.ini' to see which LLM provider is active."
echo ""
python3 server.py

echo ""
echo "================================="
echo " Server has been stopped."
echo "================================="
read -p "Press Enter to continue..."