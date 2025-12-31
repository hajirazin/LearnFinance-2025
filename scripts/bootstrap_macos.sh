#!/usr/bin/env bash
#
# bootstrap_macos.sh — One-command setup for LearnFinance-2025 on a fresh Mac.
#
# This script:
#   1. Installs Xcode Command Line Tools (may trigger interactive prompt)
#   2. Installs Nix via the Determinate Nix Installer
#   3. Installs Devbox
#   4. Runs `devbox install` to pull all repo dependencies
#
# Usage:
#   ./scripts/bootstrap_macos.sh
#
# The script is idempotent — safe to re-run if any step fails or is already done.
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Step 1: Xcode Command Line Tools
# ---------------------------------------------------------------------------
install_xcode_clt() {
    info "Checking for Xcode Command Line Tools..."

    if xcode-select -p &>/dev/null; then
        success "Xcode Command Line Tools already installed at $(xcode-select -p)"
        return 0
    fi

    info "Installing Xcode Command Line Tools..."
    info ">>> An interactive prompt will appear. Click 'Install' to proceed. <<<"

    # Trigger the install prompt
    xcode-select --install 2>/dev/null || true

    # Poll until installation completes (user must click through the prompt)
    info "Waiting for Xcode Command Line Tools installation to complete..."
    local max_wait=600  # 10 minutes max
    local waited=0
    local interval=5

    while ! xcode-select -p &>/dev/null; do
        if [ $waited -ge $max_wait ]; then
            error "Timed out waiting for Xcode CLT. Please install manually: xcode-select --install"
        fi
        sleep $interval
        waited=$((waited + interval))
        echo -n "."
    done
    echo ""

    success "Xcode Command Line Tools installed at $(xcode-select -p)"
}

# ---------------------------------------------------------------------------
# Step 2: Nix (via Determinate Nix Installer)
# ---------------------------------------------------------------------------
install_nix() {
    info "Checking for Nix..."

    if command -v nix &>/dev/null; then
        success "Nix already installed: $(nix --version)"
        return 0
    fi

    info "Installing Nix via Determinate Nix Installer..."
    info ">>> This may prompt for sudo password. <<<"

    # Determinate Nix Installer (recommended for macOS)
    # https://github.com/DeterminateSystems/nix-installer
    curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install --no-confirm

    # Source nix in current shell
    if [ -f /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]; then
        # shellcheck disable=SC1091
        . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
    fi

    if command -v nix &>/dev/null; then
        success "Nix installed: $(nix --version)"
    else
        warn "Nix installed but not in PATH. You may need to restart your shell."
        warn "Then re-run this script."
        exit 0
    fi
}

# ---------------------------------------------------------------------------
# Step 3: Devbox
# ---------------------------------------------------------------------------
install_devbox() {
    info "Checking for Devbox..."

    if command -v devbox &>/dev/null; then
        success "Devbox already installed: $(devbox version)"
        return 0
    fi

    info "Installing Devbox..."

    # Install devbox via the official installer script
    # https://www.jetify.com/devbox/docs/installing_devbox/
    curl -fsSL https://get.jetify.com/devbox | bash -s -- -f

    # The installer adds devbox to PATH via shell config, but we need it now
    export PATH="$HOME/.local/bin:$PATH"

    if command -v devbox &>/dev/null; then
        success "Devbox installed: $(devbox version)"
    else
        warn "Devbox installed but not in PATH. You may need to restart your shell."
        warn "Then re-run this script."
        exit 0
    fi
}

# ---------------------------------------------------------------------------
# Step 4: Devbox install (repo dependencies)
# ---------------------------------------------------------------------------
run_devbox_install() {
    info "Running devbox install to fetch all dependencies..."

    # Navigate to repo root (script is in scripts/)
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local repo_root
    repo_root="$(dirname "$script_dir")"

    cd "$repo_root"

    if [ ! -f "devbox.json" ]; then
        error "devbox.json not found in $repo_root"
    fi

    devbox install

    success "Devbox dependencies installed!"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo ""
    echo "=============================================="
    echo "  LearnFinance-2025 macOS Bootstrap"
    echo "=============================================="
    echo ""

    install_xcode_clt
    echo ""

    install_nix
    echo ""

    install_devbox
    echo ""

    run_devbox_install
    echo ""

    echo "=============================================="
    echo -e "${GREEN}  Bootstrap complete!${NC}"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Start a Devbox shell (recommended):"
    echo "     devbox shell"
    echo ""
    echo "  2. Start Colima (Docker runtime):"
    echo "     devbox run colima:start"
    echo ""
    echo "  3. Start n8n:"
    echo "     devbox run n8n:up"
    echo "     Open http://localhost:5678"
    echo ""
    echo "  4. Set up Brain API:"
    echo "     devbox run brain:setup"
    echo ""
    echo "  5. Run Brain API:"
    echo "     devbox run brain:run"
    echo "     Open http://localhost:8000/docs"
    echo ""
    echo "  6. Run tests:"
    echo "     devbox run brain:test"
    echo ""
    echo "See onboarding.md for full documentation."
    echo ""
}

main "$@"

