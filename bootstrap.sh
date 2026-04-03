#!/bin/bash

# Exit immediately if a command fails
set -e

echo "========================================"
echo " Smart Pen Environment Setup "
echo "========================================"

# --- 1. INSTALL SYSTEM DEPENDENCIES ---
command_exists() { command -v "$1" >/dev/null 2>&1; }

if command_exists apt; then
  echo "[+] Detected Debian/Ubuntu (apt). Installing tools..."
  sudo apt update
  sudo apt install -y build-essential cmake ninja-build libx11-dev libwayland-dev libxkbcommon-dev curl python3-venv
elif command_exists pacman; then
  echo "[+] Detected Arch Linux (pacman). Installing tools..."
  sudo pacman -Sy --needed base-devel cmake ninja wayland curl python
elif command_exists dnf; then
  echo "[+] Detected Fedora/RHEL (dnf). Installing tools..."
  sudo dnf install -y gcc-c++ cmake ninja-build libX11-devel wayland-devel curl python3
else
  echo "[-] Unsupported package manager. Please install CMake and a C++ compiler manually."
  exit 1
fi

# --- 2. INSTALL PLATFORMIO (NO PIP) ---
echo ""
echo "[+] Installing PlatformIO Core via official curl script..."
if ! command_exists pio; then
  # Download and run the isolated installer
  curl -fsSL https://raw.githubusercontent.com/platformio/platformio-core-installer/master/get-platformio.py -o get-platformio.py
  python3 get-platformio.py
  rm get-platformio.py # Clean up

  # Add PlatformIO to the user's PATH based on their active shell
  PIO_BIN_DIR="$HOME/.platformio/penv/bin"
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

# --- 3. SETUP SERIAL PERMISSIONS ---
echo ""
echo "[+] Setting up Serial Port Permissions..."

# Add user to dialout (Debian/Ubuntu) and uucp (Arch)
if getent group dialout >/dev/null 2>&1; then
  sudo usermod -aG dialout $USER
  echo "    Added $USER to 'dialout' group."
fi
if getent group uucp >/dev/null 2>&1; then
  sudo usermod -aG uucp $USER
  echo "    Added $USER to 'uucp' group."
fi

# Apply immediate fix if the device is currently plugged in
if [ -c /dev/ttyUSB0 ]; then
  sudo chmod a+rw /dev/ttyUSB0
  echo "    Granted immediate read/write access to /dev/ttyUSB0."
else
  echo "    Note: /dev/ttyUSB0 not plugged in right now. Group permissions will apply when connected."
fi

echo ""
echo "[+] Setup Complete!"
echo "    IMPORTANT 1: To use the 'pio' command immediately, run: source ~/.bashrc (or ~/.zshrc)"
echo "    IMPORTANT 2: If this is your first time, you MUST log out and log back in (or reboot) for permanent Serial port permissions."
echo "========================================"
