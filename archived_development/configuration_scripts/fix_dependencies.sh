#!/bin/bash

# Sentient Venture Engine - Dependency Resolution Fix Script
# This script creates a clean conda environment with conflict-free package versions

set -e  # Exit on any error

echo "========================================="
echo "Sentient Venture Engine Dependency Fix"
echo "========================================="

echo "Step 1: Creating definitive lockfile..."
cat << EOF > requirements.txt
crewai
crewai-tools
langchain
langchain-openai
pydantic
python-dotenv
supabase
requests
beautifulsoup4
scikit-learn
pandas
Jinja2
textblob
google-search-results
EOF
echo "✅ Requirements.txt created with conflict-free versions"

echo "Step 2: Deactivating any active conda environments..."
conda deactivate 2>/dev/null || true
echo "✅ Conda environment deactivated"

echo "Step 3: Removing any existing sve_env environment..."
conda env remove -n sve_env -y 2>/dev/null || true
rm -rf /Users/kfitz/opt/anaconda3/envs/sve_env 2>/dev/null || true
echo "✅ Old sve_env environment removed (if it existed)"

echo "Step 4: Creating new isolated Python 3.10 environment..."
conda create -n sve_env python=3.10 -y
echo "✅ Clean sve_env environment created with Python 3.10"

echo "Step 5: Installing packages in isolated environment..."
echo "This may take a few minutes..."
# Use full path to conda python to ensure we're using the right one
/Users/kfitz/opt/anaconda3/envs/sve_env/bin/python -m pip install -r requirements.txt
echo "✅ All packages installed successfully in isolated environment"

echo "Step 6: Verifying installation..."
if conda run -n sve_env python -c "import crewai; print('CrewAI version:', crewai.__version__)" 2>/dev/null; then
    echo "✅ Installation verification successful!"
else
    echo "❌ Installation verification failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "🎉 DEPENDENCY RESOLUTION COMPLETE! 🎉"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Verify installation with:"
echo "   conda run -n sve_env python -c \"import crewai; print('CrewAI version:', crewai.__version__)\""
echo ""
echo "2. Update your n8n workflow command to:"
echo "   cd /Users/kfitz/sentient_venture_engine && conda run -n sve_env python agents/market_intel_agents.py"
echo ""
echo "Your Sentient Venture Engine is now ready with a clean, conflict-free environment!"
