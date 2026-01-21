#!/bin/bash
# Publish all packages to TestPyPI
# Usage: ./scripts/publish-testpypi.sh

set -e

echo "Publishing to TestPyPI..."
echo ""
echo "IMPORTANT: Packages will be published in dependency order:"
echo "  1. llm-workers (core)"
echo "  2. llm-workers-console (depends on llm-workers)"
echo "  3. llm-workers-tools (depends on llm-workers + llm-workers-console)"
echo "  4. llm-workers-evaluation (depends on llm-workers)"
echo ""

# Confirm before proceeding
read -p "Continue with TestPyPI publish? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

# Configure TestPyPI if not already configured
echo "Ensuring TestPyPI repository is configured..."
poetry config repositories.testpypi https://test.pypi.org/legacy/ 2>/dev/null || true
echo ""

# Publish core package
echo "==> Publishing llm-workers to TestPyPI"
cd packages/llm-workers
poetry publish -r testpypi
cd ../..
echo "✓ llm-workers published to TestPyPI"
echo ""

# Wait a bit for package to be available
echo "Waiting 5 seconds for package to be available..."
sleep 5
echo ""

# Publish console package
echo "==> Publishing llm-workers-console to TestPyPI"
cd packages/llm-workers-console
poetry publish -r testpypi
cd ../..
echo "✓ llm-workers-console published to TestPyPI"
echo ""

# Wait a bit for package to be available
echo "Waiting 5 seconds for package to be available..."
sleep 5
echo ""

# Publish tools package
echo "==> Publishing llm-workers-tools to TestPyPI"
cd packages/llm-workers-tools
poetry publish -r testpypi
cd ../..
echo "✓ llm-workers-tools published to TestPyPI"
echo ""

# Wait a bit for package to be available
echo "Waiting 5 seconds for package to be available..."
sleep 5
echo ""

# Publish evaluation package
echo "==> Publishing llm-workers-evaluation to TestPyPI"
cd packages/llm-workers-evaluation
poetry publish -r testpypi
cd ../..
echo "✓ llm-workers-evaluation published to TestPyPI"
echo ""

echo "All packages published to TestPyPI successfully!"
echo ""
echo "Test installation:"
echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llm-workers-tools"
echo ""
echo "After testing, publish to production PyPI:"
echo "  ./scripts/publish-pypi.sh"
