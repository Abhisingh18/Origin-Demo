#!/bin/bash

# 🚀 AI Segmentation Studio - Quick Start Script

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🚀 AI SEGMENTATION STUDIO - QUICK START                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python
echo -e "${BLUE}[1/5]${NC} Checking Python environment..."
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python found${NC}"

# Check conda environment
echo -e "${BLUE}[2/5]${NC} Checking conda environment..."
if ! conda info --envs | grep -q "origin"; then
    echo -e "${YELLOW}⚠️  'origin' environment not found${NC}"
    echo "Creating environment..."
    # conda create -n origin python=3.10 -y
fi
echo -e "${GREEN}✅ Environment ready${NC}"

# Install requirements
echo -e "${BLUE}[3/5]${NC} Installing dependencies..."
pip install -q -r requirements.txt 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  Some dependencies may have failed${NC}"
fi

# Start backend
echo -e "${BLUE}[4/5]${NC} Starting FastAPI backend..."
echo -e "${YELLOW}📡 Backend starting on http://localhost:8000${NC}"
echo ""
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 2

# Frontend
echo -e "${BLUE}[5/5]${NC} Frontend ready!"
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    🎉 SETUP COMPLETE!                      ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo -e "║  ${GREEN}✅ Backend:  http://localhost:8000${NC}"
echo "║  📡 API Docs: http://localhost:8000/docs"
echo "║  "
echo "║  🌐 Frontend: Open in browser:"
echo "║     frontend/index.html"
echo "║  "
echo "║  Or serve with:"
echo "║     cd frontend && python -m http.server 8080"
echo "║     Open: http://localhost:8080"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${YELLOW}Press CTRL+C to stop the backend${NC}"
echo ""

# Keep script running
wait $BACKEND_PID
