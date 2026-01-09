#!/bin/bash
# Build all packages in the monorepo
# Usage: ./scripts/build-all.sh

set -e

echo "Building all packages..."
echo ""

# Build core package
echo "==> Building llm-workers (core)"
cd packages/llm-workers
poetry build
cd ../..
echo "✓ llm-workers built"
echo ""

# Build console package
echo "==> Building llm-workers-console"
cd packages/llm-workers-console
poetry build
cd ../..
echo "✓ llm-workers-console built"
echo ""

# Build tools package
echo "==> Building llm-workers-tools"
cd packages/llm-workers-tools
poetry build
cd ../..
echo "✓ llm-workers-tools built"
echo ""

echo "All packages built successfully!"
echo ""
echo "Build artifacts:"
echo "  - packages/llm-workers/dist/"
echo "  - packages/llm-workers-console/dist/"
echo "  - packages/llm-workers-tools/dist/"
echo ""
echo "Next steps:"
echo "  - Test on TestPyPI: ./scripts/publish-testpypi.sh"
echo "  - Publish to PyPI: ./scripts/publish-pypi.sh"
