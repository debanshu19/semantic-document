#!/usr/bin/env bash
# Installs and caches the real sentence-transformers embedding model AND
# the cross-encoder reranker for Semantic Document. Run from anywhere:
#
#   ./scripts/install-embeddings.sh
#
# This is a one-time setup step. Once it finishes, the app runs both
# models fully offline forever after (see app/embeddings.py and
# app/reranker.py) -- that's the whole point of the app's privacy
# model, not just a workaround.
#
# On the Walmart corporate network, two internal mirrors make this work
# without ever touching the public internet:
#   - PyPI package:  the mlplatforms-pypi / external-pypi Artifactory mirrors
#   - Model weights: the internal Hugging Face Hub Artifactory mirror
# Both sit behind the corporate TLS-inspecting proxy, so we also need to
# trust the Walmart Root/Intermediate CAs (exported from the System
# keychain) alongside the normal public CA bundle.
set -euo pipefail

cd "$(dirname "$0")/.."

WALMART_PYPI_INDEX="https://pypi.ci.artifacts.walmart.com/artifactory/api/pypi/mlplatforms-pypi/simple"
WALMART_HOST="pypi.ci.artifacts.walmart.com"
HF_MIRROR="https://ci.artifacts.walmart.com/artifactory/api/huggingfaceml/hub-huggingfaceml-release-remote"
CA_BUNDLE="${TMPDIR:-/tmp}/semantic-document-ca-bundle.pem"

echo "=============================================================="
echo " Step 1/3: Installing the sentence-transformers package"
echo "=============================================================="
echo "(this also provides CrossEncoder, used for reranking -- no"
echo " separate package needed)"
echo ""

install_with() {
    local index_url="$1"
    local host
    host=$(echo "$index_url" | sed -E 's#https?://([^/]+).*#\1#')
    if command -v uv >/dev/null 2>&1; then
        uv pip install sentence-transformers --index-url "$index_url" --allow-insecure-host "$host"
    else
        pip install sentence-transformers --index-url "$index_url"
    fi
}

if ! install_with "$WALMART_PYPI_INDEX"; then
    echo "--> Walmart mirror failed. Trying public PyPI (works off VPN)..."
    install_with "https://pypi.org/simple"
fi

echo ""
echo "=============================================================="
echo " Step 2/3: Trusting the corporate proxy's TLS certificates"
echo "=============================================================="
echo "(needed because the proxy re-signs HTTPS traffic with Walmart's"
echo " own certificate authority, which Python doesn't trust by default)"
echo ""

if command -v security >/dev/null 2>&1; then
    PY_CACERT=$(uv run python -c "import certifi; print(certifi.where())" 2>/dev/null || python3 -c "import certifi; print(certifi.where())")
    cp "$PY_CACERT" "$CA_BUNDLE"
    for cert in "WalmartRootCA-SHA256" "WalmartIntermediateCA01-SHA256" "WalmartIssuingCA-2FA-01-SHA256" "WalmartIssuingCA-2FA-02-SHA256"; do
        security find-certificate -c "$cert" -p /Library/Keychains/System.keychain 2>/dev/null >> "$CA_BUNDLE" || true
    done
    echo "Combined CA bundle written to $CA_BUNDLE"
else
    echo "Not on macOS (no 'security' tool) -- skipping. If the next step"
    echo "fails with a certificate error, ask your platform team for the"
    echo "corporate root CA in PEM form and set REQUESTS_CA_BUNDLE/SSL_CERT_FILE"
    echo "to point at it."
    CA_BUNDLE=""
fi

echo ""
echo "=============================================================="
echo " Step 3/3: Downloading and caching both models (one-time)"
echo "=============================================================="
echo "Using the internal Hugging Face mirror: $HF_MIRROR"
echo "Embedding model: sentence-transformers/all-MiniLM-L6-v2 (~80MB)"
echo "Reranker model:  cross-encoder/ms-marco-MiniLM-L-6-v2 (~90MB)"
echo ""

HF_ENDPOINT="$HF_MIRROR" \
REQUESTS_CA_BUNDLE="$CA_BUNDLE" \
SSL_CERT_FILE="$CA_BUNDLE" \
uv run python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
vec = model.encode('hello world')
print(f'Embedding model OK -- dimension: {len(vec)}')

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
score = reranker.predict([('hello', 'world')])
print(f'Reranker model OK -- sample score: {score}')
"

echo ""
echo "=============================================================="
echo " Verifying the app's own modules work fully offline from here"
echo "=============================================================="
uv run python -c "
from app.embeddings import embed_texts, embed_query
from app.reranker import rerank
import numpy as np

vecs = embed_texts(['the quick brown fox', 'a lazy sleepy dog'])
q = embed_query('quick fox')
print('Cosine sim to fox sentence (should be high):', float(np.dot(q, vecs[0])))
print('Cosine sim to dog sentence (should be low):', float(np.dot(q, vecs[1])))

scores = rerank('what are the database choices', ['For storage, choose between a real-time store and a durable store.', 'The weather today is sunny with a light breeze.'])
print('Rerank scores (first should be higher):', scores)
"

echo ""
echo "Done. Both models are cached at ~/.cache/huggingface and the app"
echo "runs fully offline from here -- restart it and search/finalize"
echo "will use real semantic embeddings plus cross-encoder reranking."
echo ""
echo "If step 1 or 3 hangs instead of failing outright:"
echo "  1. Try again off VPN / on a different network (home wifi, hotspot)."
echo "  2. Request an allowlist exception for blob.core.windows.net at"
echo "     https://puppy.walmart.com/url-allowlist (auto-approved, ~5 min)."
echo "  3. On a machine that CAN reach it, run:"
echo "       pip download sentence-transformers -d ./wheels"
echo "     then copy the wheels/ folder here and run:"
echo "       uv pip install --no-index --find-links ./wheels sentence-transformers"
