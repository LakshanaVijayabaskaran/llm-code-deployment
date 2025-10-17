#!/usr/bin/env bash
set -e

# Install GitHub CLI, TruffleHog, and Gitleaks
apt-get update
apt-get install -y curl git
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
apt-get update
apt-get install -y gh

# Install secret scanners (optional)
pip install trufflehog gitleaks

echo "✅ Environment ready."
