#!/bin/bash
# Set version for all packages in the monorepo
# Usage: ./scripts/set-version.sh 1.0.0-rc9

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <version>"
  echo "Example: $0 1.0.0-rc9"
  exit 1
fi

VERSION=$1

echo "Setting version to $VERSION..."

# Update version in root pyproject.toml
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
rm pyproject.toml.bak

# Update version in llm-workers
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" packages/llm-workers/pyproject.toml
rm packages/llm-workers/pyproject.toml.bak

# Update version in llm-workers-console
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" packages/llm-workers-console/pyproject.toml
# Update llm-workers dependency in llm-workers-console
sed -i.bak "s/\"llm-workers (==.*)\"/\"llm-workers (==$VERSION)\"/" packages/llm-workers-console/pyproject.toml
rm packages/llm-workers-console/pyproject.toml.bak

# Update version in llm-workers-tools
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" packages/llm-workers-tools/pyproject.toml
# Update llm-workers dependency in llm-workers-tools
sed -i.bak "s/\"llm-workers (==.*)\"/\"llm-workers (==$VERSION)\"/" packages/llm-workers-tools/pyproject.toml
# Update llm-workers-console dependency in llm-workers-tools
sed -i.bak "s/\"llm-workers-console (==.*)\"/\"llm-workers-console (==$VERSION)\"/" packages/llm-workers-tools/pyproject.toml
rm packages/llm-workers-tools/pyproject.toml.bak

echo "✓ Version updated to $VERSION in all packages"
echo ""
echo "Changed files:"
echo "  - pyproject.toml"
echo "  - packages/llm-workers/pyproject.toml"
echo "  - packages/llm-workers-console/pyproject.toml"
echo "  - packages/llm-workers-tools/pyproject.toml"
echo ""
echo "Next steps:"
echo "  1. Update docs/release-notes.md with release notes for version $VERSION"
echo "  2. Review changes: git diff"
echo "  3. Commit: git add -A && git commit -m 'Bump version to $VERSION'"
echo "  4. Build: ./scripts/build-all.sh"
echo "  5. Publish: ./scripts/publish-testpypi.sh (test) or ./scripts/publish-pypi.sh (production)"
