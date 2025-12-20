#!/bin/bash

# Agglomerative Clustering Demo - Run Script

echo "=================================="
echo "Agglomerative Clustering Demo"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Tạo virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Kích hoạt virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Cài đặt dependencies..."
pip install -q -r requirements.txt

# Run app
echo ""
echo "🚀 Khởi chạy ứng dụng..."
echo "➡️  Mở browser tại: http://localhost:8501"
echo ""

streamlit run app.py
