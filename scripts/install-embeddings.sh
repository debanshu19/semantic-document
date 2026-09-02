#!/usr/bin/env bash
# Installs the real sentence-transformers embedding model for Semantic
# Document. Run from anywhere:
#
#   ./scripts/install-embeddings.sh
#
# This is a one-time step -- once installed, the model itself downloads
# from Hugging Face on first use (~80MB) and everything runs fully
# offline after that.
set -euo pipefail

cd "$(dirname "$0")/.."

WALMART_INDEX="https://pypi.ci.artifacts.walmart.com/artifactory/api/pypi/external-pypi/simple"
WALMART_HOST="pypi.ci.artifacts.walmart.com"

echo "=============================================================="
echo " Installing sentence-transformers"
echo "=============================================================="
echo ""
echo "Heads up: this pulls in torch, a large (~200MB+) download. On"
echo "some corporate networks/VPNs, the internal package mirror"
echo "redirects large wheels to Azure Blob Storage, and that specific"
echo "domain can be blocked -- if this hangs for more than 2-3 minutes"
echo "with your terminal doing nothing, that's almost certainly what's"
echo "happening. Ctrl+C and see the 'If it hangs' section below."
echo ""

install_with() {
    local index_url="$1"
    local host
    host=$(echo "$index_url" | sed -E 's#https?://([^/]+).*#\1#')

    if command -v uv >/dev/null 2>&1; then
        uv pip install sentence-transformers \
            --index-url "$index_url" \
            --allow-insecure-host "$host"
    else
        pip install sentence-transformers --index-url "$index_url"
    fi
}

echo "--> Trying the Walmart internal mirror first..."
if install_with "$WALMART_INDEX"; then
    echo "Installed via internal mirror."
else
    echo ""
    echo "--> Internal mirror failed or was interrupted. Trying public PyPI"
    echo "    directly (works if you're off VPN / on an unrestricted network)..."
    install_with "https://pypi.org/simple"
fi

echo ""
echo "=============================================================="
echo " Verifying it actually works"
echo "=============================================================="
echo "(this downloads the ~80MB model from Hugging Face once)"
echo ""

if command -v uv >/dev/null 2>&1; then
    uv run python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
vec = model.encode('hello world')
print(f'Success -- embedding dimension: {len(vec)}')
"
else
    python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
vec = model.encode('hello world')
print(f'Success -- embedding dimension: {len(vec)}')
"
fi

echo ""
echo "Done. Restart the app and Finalize should work with real embeddings now."
echo ""
echo "If it hangs:"
echo "  1. Try again off VPN / on a different network (home wifi, hotspot)."
echo "  2. Request an allowlist exception for blob.core.windows.net at"
echo "     https://puppy.walmart.com/url-allowlist (auto-approved, ~5 min)."
echo "  3. On a machine that CAN reach it, run:"
echo "       pip download sentence-transformers -d ./wheels"
echo "     then copy the wheels/ folder here and run:"
echo "       uv pip install --no-index --find-links ./wheels sentence-transformers"
