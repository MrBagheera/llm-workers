#!/bin/bash
# Publish all packages to PyPI (production)
# Usage: ./scripts/publish-pypi.sh

set -e

echo "======================================"
echo "WARNING: Publishing to Production PyPI"
echo "======================================"
echo ""
echo "This will publish packages to the production PyPI repository."
echo "Make sure you have:"
echo "  1. Updated the version with ./scripts/set-version.sh"
echo "  2. Updated docs/release-notes.md with release notes"
echo "  3. Built all packages with ./scripts/build-all.sh"
echo "  4. Tested on TestPyPI with ./scripts/publish-testpypi.sh"
echo "  5. Committed all changes to git"
echo ""
echo "Packages will be published in dependency order:"
echo "  1. llm-workers (core)"
echo "  2. llm-workers-console (depends on llm-workers)"
echo "  3. llm-workers-tools (depends on llm-workers + llm-workers-console)"
echo "  4. llm-workers-evaluation (depends on llm-workers)"
echo ""

# Confirm before proceeding
read -p "Are you sure you want to publish to production PyPI? (yes/N) " -r
echo
if [[ ! $REPLY =~ ^yes$ ]]; then
  echo "Aborted. Type 'yes' to confirm."
  exit 1
fi

# Publish core package
echo "==> Publishing llm-workers to PyPI"
cd packages/llm-workers
poetry publish
cd ../..
echo "✓ llm-workers published to PyPI"
echo ""

# Wait a bit for package to be available
echo "Waiting 10 seconds for package to be available..."
sleep 10
echo ""

# Publish console package
echo "==> Publishing llm-workers-console to PyPI"
cd packages/llm-workers-console
poetry publish
cd ../..
echo "✓ llm-workers-console published to PyPI"
echo ""

# Wait a bit for package to be available
echo "Waiting 10 seconds for package to be available..."
sleep 10
echo ""

# Publish tools package
echo "==> Publishing llm-workers-tools to PyPI"
cd packages/llm-workers-tools
poetry publish
cd ../..
echo "✓ llm-workers-tools published to PyPI"
echo ""

# Wait a bit for package to be available
echo "Waiting 10 seconds for package to be available..."
sleep 10
echo ""

# Publish evaluation package
echo "==> Publishing llm-workers-evaluation to PyPI"
cd packages/llm-workers-evaluation
poetry publish
cd ../..
echo "✓ llm-workers-evaluation published to PyPI"
echo ""

echo "======================================"
echo "All packages published to PyPI successfully!"
echo "======================================"
echo ""
echo "Post-release steps:"
echo "  1. Create a git tag: git tag v\$(poetry version -s) && git push --tags"
echo "  2. Create a GitHub release with the release notes"
echo "  3. Announce the release"
