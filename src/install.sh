#!/usr/bin/env bash
# ============================================================================
# UCS Installer — Unified Cognitive Substrate
# Architecture: William Kyle Million (~K¹), IntuiTek¹
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-$HOME/.ucs}"

# Colors (if terminal supports them)
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[UCS]${NC} $1"; }
ok()    { echo -e "${GREEN}[UCS]${NC} $1"; }
err()   { echo -e "${RED}[UCS]${NC} $1" >&2; }
bold()  { echo -e "${BOLD}$1${NC}"; }

# ---- Header ----
echo ""
bold "╔══════════════════════════════════════════════════════════════╗"
bold "║        UNIFIED COGNITIVE SUBSTRATE (UCS) v1.1 — Installer        ║"
bold "║     Architecture: William Kyle Million (~K¹), IntuiTek¹      ║"
bold "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ---- Check Python ----
info "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "python3 is required (Python >= 3.10). Install python3."
    exit 1
fi
PYTHON="python3"
else
    PYTHON="python3"
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    err "Python >= 3.10 required. Found: $PY_VERSION"
    exit 1
fi
ok "Python $PY_VERSION found."

# ---- Verify source files ----
info "Verifying source files..."
REQUIRED_FILES=("bridge.py" "torusfield_kernel.py" "SKILL.md")
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$f" ]; then
        err "Missing required file: $f"
        exit 1
    fi
done
ok "All source files present."

# ---- Detect platform ----
SKILL_DIR=""
if [ -d "$HOME/.openclaw" ]; then
    SKILL_DIR="$HOME/.openclaw/workspace/skills/ucs"
    info "Detected OpenClaw environment."
elif [ -d ".claude" ]; then
    SKILL_DIR=".claude/skills/ucs"
    info "Detected Claude Code environment."
else
    SKILL_DIR="$HOME/.ucs/skill"
    info "No known agent platform detected. Installing to $SKILL_DIR"
fi

# ---- Install skill files ----
info "Installing skill files to: $SKILL_DIR"
mkdir -p "$SKILL_DIR"
cp "$SCRIPT_DIR/bridge.py" "$SKILL_DIR/"
cp "$SCRIPT_DIR/torusfield_kernel.py" "$SKILL_DIR/"
cp "$SCRIPT_DIR/SKILL.md" "$SKILL_DIR/"

# Copy documentation alongside
for doc in ATTRIBUTION.md LICENSE READ_ME_FIRST.md BUILD_PLAN.md; do
    if [ -f "$SCRIPT_DIR/$doc" ]; then
        cp "$SCRIPT_DIR/$doc" "$SKILL_DIR/"
    fi
done
ok "Skill files installed."

# ---- Initialize workspace ----
info "Initializing workspace at: $WORKSPACE"
INIT_OUTPUT=$($PYTHON "$SKILL_DIR/bridge.py" --workspace "$WORKSPACE" init 2>&1)
INIT_STATUS=$(echo "$INIT_OUTPUT" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "error")

if [ "$INIT_STATUS" = "initialized" ]; then
    ok "Workspace initialized successfully."
elif [ "$INIT_STATUS" = "already_initialized" ]; then
    ok "Workspace already exists (use --force to reinitialize)."
else
    err "Initialization returned unexpected status: $INIT_STATUS"
    echo "$INIT_OUTPUT"
    exit 1
fi

# ---- Verify ----
info "Running verification..."
STATUS_OUTPUT=$($PYTHON "$SKILL_DIR/bridge.py" --workspace "$WORKSPACE" status 2>&1)
STATUS_OK=$(echo "$STATUS_OUTPUT" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "error")

if [ "$STATUS_OK" = "ok" ] || [ "$STATUS_OK" = "not_initialized" ]; then
    ok "Verification passed."
else
    err "Verification failed."
    echo "$STATUS_OUTPUT"
    exit 1
fi

# ---- Summary ----
echo ""
bold "════════════════════════════════════════════════════════════════"
ok "UCS installed successfully."
echo ""
info "Skill directory:  $SKILL_DIR"
info "Workspace:        $WORKSPACE"
echo ""
info "Quick commands:"
echo "  $PYTHON $SKILL_DIR/bridge.py --workspace $WORKSPACE status"
echo "  $PYTHON $SKILL_DIR/bridge.py --workspace $WORKSPACE resume"
echo "  $PYTHON $SKILL_DIR/bridge.py --workspace $WORKSPACE consult --context \"your task\""
echo ""
info "Read SKILL.md for full operational instructions."
bold "════════════════════════════════════════════════════════════════"
echo ""
