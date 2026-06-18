#!/usr/bin/env bash
# =============================================================================
# install_mac.sh
# macOS Apple Silicon (M1/M2/M3/M4) installer for Xenium Sample PCA
# =============================================================================
# Run once from the project root:
#   chmod +x install_mac.sh && ./install_mac.sh
#
# What this does:
#   1.  Ensures Xcode Command Line Tools are present
#   2.  Installs Miniforge3 (ARM64 conda) directly via curl — no Homebrew needed
#   3.  Creates the conda environment: xenium_sample_pca  (Python 3.11)
#   4.  Installs the (small) dependency set via conda-forge + pip
#   5.  Verifies the installation
#   6.  Sets the macOS matplotlib backend
# =============================================================================

# Do NOT use "set -e" here — we handle errors explicitly so the script
# never dies silently mid-way through a long package install.

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; BOLD='\033[1m'; RESET='\033[0m'

ENV_NAME="xenium_sample_pca"
PYTHON_VERSION="3.11"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
MINIFORGE_INSTALLER="/tmp/Miniforge3-arm64.sh"

log()  { echo -e "${CYAN}[install]${RESET} $*"; }
ok()   { echo -e "${GREEN}[  ok  ]${RESET} $*"; }
warn() { echo -e "${YELLOW}[ warn ]${RESET} $*"; }
fail() { echo -e "${RED}[ FAIL ]${RESET} $*"; echo ""; echo "Installation stopped. Fix the error above and re-run."; exit 1; }
sep()  { echo -e "${BOLD}────────────────────────────────────────────────${RESET}"; }

sep
echo -e "${BOLD}  Xenium Sample PCA — macOS Installer${RESET}"
echo -e "  Environment : ${ENV_NAME}  |  Python ${PYTHON_VERSION}"
sep
echo ""

# =============================================================================
# 0. Sanity checks
# =============================================================================
ARCH=$(uname -m)
OS=$(uname -s)

[[ "$OS" == "Darwin" ]] || fail "This installer is for macOS only (detected: $OS)."

if [[ "$ARCH" != "arm64" ]]; then
    warn "Expected arm64 (Apple Silicon), detected: $ARCH"
    warn "The installer will continue but some packages may not be ARM64-native."
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
    MINIFORGE_INSTALLER="/tmp/Miniforge3-x86_64.sh"
else
    ok "Architecture: arm64 (Apple Silicon M-series)"
fi

# =============================================================================
# 1. Xcode Command Line Tools
# =============================================================================
log "Checking Xcode Command Line Tools …"
if ! xcode-select -p &>/dev/null; then
    log "Installing Xcode Command Line Tools (this may open a dialog — click Install) …"
    xcode-select --install 2>/dev/null || true
    log "Waiting for Xcode CLT installation to finish …"
    until xcode-select -p &>/dev/null; do
        sleep 5
        echo -n "."
    done
    echo ""
    ok "Xcode Command Line Tools installed."
else
    ok "Xcode Command Line Tools: $(xcode-select -p)"
fi

# =============================================================================
# 2. Find or install conda (Miniforge3)
#    We download directly — no Homebrew dependency.
# =============================================================================
log "Checking for conda / Miniforge3 …"

CONDA_PATH=""
for candidate in \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/mambaforge/bin/conda" \
    "$HOME/opt/miniforge3/bin/conda" \
    "/opt/miniforge3/bin/conda" \
    "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
    "/usr/local/miniforge3/bin/conda" \
    "/usr/local/bin/conda"
do
    if [[ -x "$candidate" ]]; then
        CONDA_PATH="$candidate"
        break
    fi
done

if [[ -z "$CONDA_PATH" ]]; then
    log "Conda not found. Downloading Miniforge3 (ARM64) …"

    if ! curl -fsSL -o "$MINIFORGE_INSTALLER" "$MINIFORGE_URL"; then
        fail "Download failed. Check your internet connection and try again."
    fi
    ok "Downloaded: $MINIFORGE_INSTALLER ($(du -sh "$MINIFORGE_INSTALLER" | cut -f1))"

    log "Installing Miniforge3 to $HOME/miniforge3 …"
    bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3" \
        || fail "Miniforge3 installation failed (exit code $?)."
    rm -f "$MINIFORGE_INSTALLER"

    CONDA_PATH="$HOME/miniforge3/bin/conda"
    [[ -x "$CONDA_PATH" ]] || fail "Miniforge3 install finished but conda not found at $CONDA_PATH"
    ok "Miniforge3 installed: $CONDA_PATH"
else
    ok "Conda found: $CONDA_PATH"
fi

# Add conda to PATH for the rest of this script
CONDA_BASE=$("$CONDA_PATH" info --base 2>/dev/null) \
    || fail "Could not determine conda base directory from $CONDA_PATH"

source "$CONDA_BASE/etc/profile.d/conda.sh" \
    || fail "Could not source conda shell functions from $CONDA_BASE"

ok "Conda base: $CONDA_BASE"
ok "Conda version: $(conda --version)"

# =============================================================================
# 3. Create / update the conda environment
# =============================================================================
sep
log "Setting up conda environment '${ENV_NAME}' …"

if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    warn "Environment '${ENV_NAME}' already exists."
    echo ""
    read -r -p "  Remove and recreate from scratch? [y/N]  " ans
    echo ""
    if [[ "$ans" =~ ^[Yy]$ ]]; then
        log "Removing old environment …"
        conda env remove -n "$ENV_NAME" -y \
            || fail "Could not remove existing environment."
        ok "Old environment removed."
    else
        log "Keeping existing environment and updating packages."
    fi
fi

if ! conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    log "Creating environment (Python ${PYTHON_VERSION}) …"
    conda create -n "$ENV_NAME" python="${PYTHON_VERSION}" -c conda-forge -y \
        || fail "Could not create conda environment."
fi
ok "Environment '${ENV_NAME}' ready."

# =============================================================================
# 4. Activate and install packages
# =============================================================================
sep
log "Activating '${ENV_NAME}' …"
conda activate "$ENV_NAME" \
    || fail "Could not activate environment '${ENV_NAME}'."

ACTIVE_PY=$(python --version 2>&1)
ok "Active Python: $ACTIVE_PY  |  $(which python)"

# --- 4a. Core scientific stack (conda-forge — best ARM64 builds) -------------
log "Installing core scientific stack via conda-forge …"
conda install -n "$ENV_NAME" -c conda-forge -y \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    anndata \
    pyarrow \
    h5py \
    hdf5 \
    || fail "conda-forge core install failed."
ok "Core scientific stack installed."

# --- 4b. Web interface (pip) -------------------------------------------------
log "Installing the web interface (streamlit, plotly) …"
pip install streamlit plotly --quiet \
    || fail "pip install failed."
ok "Web interface installed."

# --- 4c. Leiden Optimizer stack (single-cell clustering) ---------------------
# Only the 🔎 Leiden Optimizer page needs these; the rest of the app runs
# without them. A failure here is non-fatal so the core install still succeeds.
log "Installing the Leiden Optimizer stack (scanpy, igraph, leidenalg, harmonypy) …"
if conda install -n "$ENV_NAME" -c conda-forge -y \
    scanpy \
    python-igraph \
    leidenalg \
    harmonypy ; then
    ok "Leiden Optimizer stack installed."
else
    warn "Could not install the Leiden Optimizer stack. The Sample-PCA workflow"
    warn "still works; to enable step 4 later, run:"
    warn "  conda install -n ${ENV_NAME} -c conda-forge scanpy python-igraph leidenalg harmonypy"
fi

# --- 4d. Install the project itself (editable) -------------------------------
# Installs the xenium_spatial package so `import xenium_spatial` works from
# anywhere (the dependencies above are already satisfied, so --no-deps is safe
# and fast).
log "Installing the xenium_spatial package (editable) …"
pip install -e . --no-deps --quiet \
    && ok "xenium_spatial installed (editable)." \
    || warn "Editable install failed; the app still works via its src/ path shim."

# =============================================================================
# 5. Verification
# =============================================================================
sep
log "Verifying installation …"

python - << 'PYCHECK'
import sys
failures = []

required = [
    ("numpy",        "numpy"),
    ("pandas",       "pandas"),
    ("scipy",        "scipy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib",   "matplotlib"),
    ("anndata",      "anndata"),
    ("pyarrow",      "pyarrow"),
    ("streamlit",    "streamlit"),
    ("plotly",       "plotly"),
]

for label, mod in required:
    try:
        __import__(mod)
        print(f"  \033[32m✓\033[0m  {label}")
    except ImportError as e:
        print(f"  \033[31m✗\033[0m  {label}  —  {e}")
        failures.append(label)

optional = [
    ("scanpy",    "scanpy"),
    ("igraph",    "igraph"),
    ("leidenalg", "leidenalg"),
    ("harmonypy", "harmonypy"),
]
print("\n  Leiden Optimizer stack (optional — step 4 only):")
for label, mod in optional:
    try:
        __import__(mod)
        print(f"  \033[32m✓\033[0m  {label}")
    except ImportError:
        print(f"  \033[33m–\033[0m  {label}  (not installed; the Leiden Optimizer page needs it)")

print("")
if failures:
    print(f"  \033[31mFailed packages: {failures}\033[0m")
    sys.exit(1)
else:
    print("  \033[32mAll required packages verified.\033[0m")
PYCHECK

[[ $? -eq 0 ]] || fail "One or more required packages failed to import."

# =============================================================================
# 6. matplotlib backend for macOS
# =============================================================================
log "Configuring matplotlib backend …"
MATPLOTLIBRC="$HOME/.matplotlib/matplotlibrc"
mkdir -p "$(dirname "$MATPLOTLIBRC")"
if grep -q "^backend" "$MATPLOTLIBRC" 2>/dev/null; then
    ok "matplotlib backend already set in $MATPLOTLIBRC"
else
    echo "backend : MacOSX" >> "$MATPLOTLIBRC"
    ok "Set  backend: MacOSX  in $MATPLOTLIBRC"
fi

# =============================================================================
# 7. Make start_app.command executable
# =============================================================================
if [[ -f "start_app.command" ]]; then
    chmod +x start_app.command
    ok "start_app.command is now executable (double-click to launch)."
fi

# =============================================================================
# 8. Add conda init to shell profile (so 'conda activate' works in new tabs)
# =============================================================================
log "Ensuring conda initialised in shell profile …"
"$CONDA_PATH" init zsh  2>/dev/null || true
"$CONDA_PATH" init bash 2>/dev/null || true
ok "Shell profile updated (takes effect in new Terminal tabs)."

# =============================================================================
# Done
# =============================================================================
sep
echo -e "${GREEN}${BOLD}  Installation complete!${RESET}"
sep
echo ""
echo -e "  ${BOLD}To launch the web interface (easiest):${RESET}"
echo -e "    Double-click  ${CYAN}start_app.command${RESET}  in Finder"
echo -e "    — or —"
echo -e "    ${CYAN}conda activate ${ENV_NAME}${RESET}"
echo -e "    ${CYAN}streamlit run app/app.py${RESET}"
echo ""
echo -e "  ${BOLD}To run the sample PCA from the command line:${RESET}"
echo -e "    ${CYAN}conda activate ${ENV_NAME} && python scripts/run_sample_pca.py${RESET}"
echo ""
echo -e "  ${BOLD}Note:${RESET} Open a new Terminal tab before running — the"
echo -e "  'conda activate' command needs the updated shell profile."
echo ""
