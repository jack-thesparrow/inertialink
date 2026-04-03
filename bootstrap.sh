#!/bin/bash

# Exit immediately if a command fails
set -e

echo "========================================"
echo " Smart Pen Environment Setup"
echo "========================================"

# --- 1. INSTALL SYSTEM DEPENDENCIES ---
command_exists() { command -v "$1" >/dev/null 2>&1; }

if command_exists apt; then
  echo "[+] Detected Debian/Ubuntu (apt). Checking tools..."
  sudo apt update
  sudo apt install -y build-essential cmake ninja-build libx11-dev libwayland-dev libxkbcommon-dev curl python3-venv
elif command_exists pacman; then
  echo "[+] Detected Arch Linux (pacman). Checking tools..."
  sudo pacman -Sy --needed base-devel cmake ninja wayland curl python
elif command_exists dnf; then
  echo "[+] Detected Fedora/RHEL (dnf). Checking tools..."
  sudo dnf install -y gcc-c++ cmake ninja-build libX11-devel wayland-devel curl python3
else
  echo "[-] Unsupported package manager. Please install CMake and a C++ compiler manually."
  exit 1
fi

# --- 2. PYTHON VIRTUAL ENVIRONMENT ---
echo ""
echo "[+] Setting up Python Virtual Environment..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "    Created .venv folder."
else
  echo "    .venv already exists."
fi

echo "[+] Installing Python dependencies from requirements.txt..."
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
echo "    Dependencies installed."

# --- 3. INSTALL PLATFORMIO (NO PIP) ---
echo ""
echo "[+] Checking PlatformIO Core..."
PIO_BIN_DIR="$HOME/.platformio/penv/bin"

if ! command_exists pio; then
  echo "    Installing PlatformIO via official curl script..."
  curl -fsSL https://raw.githubusercontent.com/platformio/platformio-core-installer/master/get-platformio.py -o get-platformio.py
  python3 get-platformio.py
  rm get-platformio.py

  echo ""
  echo "[+] Adding PlatformIO to your PATH..."
  if [ -n "$ZSH_VERSION" ]; then
    PROFILE_FILE="$HOME/.zshrc"
  else
    PROFILE_FILE="$HOME/.bashrc"
  fi

  if ! grep -q "$PIO_BIN_DIR" "$PROFILE_FILE"; then
    echo "export PATH=\"\$PATH:$PIO_BIN_DIR\"" >>"$PROFILE_FILE"
    echo "    Appended PATH to $PROFILE_FILE"
  else
    echo "    PATH already configured in $PROFILE_FILE"
  fi
else
  echo "    PlatformIO is already installed!"
fi

# --- 4. SETUP SERIAL PERMISSIONS ---
echo ""
echo "[+] Setting up Serial Port Permissions..."
if getent group dialout >/dev/null 2>&1; then
  sudo usermod -aG dialout $USER
  echo "    Added $USER to 'dialout' group."
fi
if getent group uucp >/dev/null 2>&1; then
  sudo usermod -aG uucp $USER
  echo "    Added $USER to 'uucp' group."
fi

PEN_PLUGGED_IN=false
if [ -c /dev/ttyUSB0 ]; then
  sudo chmod a+rw /dev/ttyUSB0
  PEN_PLUGGED_IN=true
  echo "    Granted immediate read/write access to /dev/ttyUSB0."
fi

# --- 5. NEXT STEPS & TROUBLESHOOTING SUMMARY ---
echo ""
echo "========================================"
echo " SETUP COMPLETE - NEXT STEPS"
echo "========================================"
echo ""
echo "[+] ENVIRONMENT ACTIVATION:"
echo "    To begin development, manually activate your tools:"
echo "    1. Python Sandbox:  source .venv/bin/activate"
echo "    2. PlatformIO CLI:  source ~/.bashrc (or ~/.zshrc)"
echo ""
echo "--- COMMON ISSUES ---"

# Dynamic hardware warning
if [ "$PEN_PLUGGED_IN" = false ]; then
  echo "[!] SERIAL PORT ERROR: 'Failed to open /dev/ttyUSB0 (Check permissions!)'"
  echo "    Reason: The pen was not plugged in during setup, so immediate"
  echo "    permissions could not be granted."
  echo "    Fix: You MUST log out of Linux and log back in (or reboot)"
  echo "    for your new hardware group permissions to apply permanently."
  echo ""
else
  echo "[*] SERIAL PORT: Immediate permissions granted."
  echo "    Note: For permanent access across reboots, your new user group"
  echo "    permissions will apply the next time you log out and log back in."
  echo ""
fi

echo "[!] PYTHON ERROR: 'ModuleNotFoundError: No module named pandas/numpy'"
echo "    Reason: You are using the global Linux Python instead of the sandbox."
echo "    Fix: Run 'source .venv/bin/activate' before running Python scripts."
echo ""
echo "[!] CMAKE ERROR: 'Could not find compile_commands.json'"
echo "    Reason: The Neovim symlink hasn't been generated yet."
echo "    Fix: Run 'cmake -B build/' to generate it."
echo "========================================"
