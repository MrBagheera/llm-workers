#!/bin/bash
# Clean all build artifacts
# Usage: ./scripts/clean-build.sh

set -e

echo "Cleaning build artifacts..."

# Clean core package
if [ -d "packages/llm-workers/dist" ]; then
  rm -rf packages/llm-workers/dist
  echo "✓ Cleaned packages/llm-workers/dist"
fi

# Clean console package
if [ -d "packages/llm-workers-console/dist" ]; then
  rm -rf packages/llm-workers-console/dist
  echo "✓ Cleaned packages/llm-workers-console/dist"
fi

# Clean tools package
if [ -d "packages/llm-workers-tools/dist" ]; then
  rm -rf packages/llm-workers-tools/dist
  echo "✓ Cleaned packages/llm-workers-tools/dist"
fi

# Clean evaluation package
if [ -d "packages/llm-workers-evaluation/dist" ]; then
  rm -rf packages/llm-workers-evaluation/dist
  echo "✓ Cleaned packages/llm-workers-evaluation/dist"
fi

echo ""
echo "Build artifacts cleaned!"
