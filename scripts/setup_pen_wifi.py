"""
InertiaLink WiFi setup — SoftAP mode (no router required).

The ESP32 always runs its own access point:
  SSID     : InertiaLink
  Password : inertia123
  ESP32 IP : 192.168.4.1

This script sends MODE:WIFI over USB to tell the pen to start streaming
over its SoftAP instead of the USB serial port.  Then it prints the
connection instructions for the laptop side.
"""

import serial
import serial.tools.list_ports
import time

AP_SSID = "InertiaLink"
AP_PASS = "inertia123"


def find_esp32_port():
    print("Scanning for ESP32 USB ports...")
    ports = serial.tools.list_ports.comports()

    for port in ports:
        desc = str(port.description).lower()
        if 'cp210' in desc or 'ch340' in desc or 'silicon' in desc or 'wch' in desc:
            return port.device

    # Fallback to first available ttyUSB/ttyACM
    for port in ports:
        if 'ttyUSB' in port.device or 'ttyACM' in port.device:
            return port.device

    return None


def main():
    print("=== InertiaLink Smart Pen WiFi Setup (SoftAP) ===\n")
    print("The pen creates its own WiFi network — no router needed.\n")

    port = find_esp32_port()
    if not port:
        print("ERROR: Could not find the ESP32 USB port.")
        print("       Plug the pen into this PC, then run this script again.")
        return

    print(f"Detected pen on: {port}\n")

    try:
        ser = serial.Serial(port, 115200, timeout=0.2)
        ser.setDTR(False)
        ser.setRTS(False)

        print("[Sys] Waiting for ESP32 to finish booting (up to 10 s)...")
        ready = False
        start = time.time()
        while time.time() - start < 15.0:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"  [Boot] {line}")
            if "[ESP] Ready" in line:
                ready = True
                break

        if not ready:
            print("\n[WARNING] Never saw the 'Ready' signal — sending command anyway.")

        time.sleep(0.3)
        ser.reset_input_buffer()

        print("\n[Sys] Switching pen to WiFi (SoftAP) mode...")
        ser.write(b"MODE:WIFI\n")
        ser.flush()

        success = False
        start = time.time()
        while time.time() - start < 5.0:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"  > {line}")
            if "[ESP] WiFi OK" in line:
                success = True
                break
            if "[ESP] WiFi failed" in line:
                break

        ser.close()

        if success:
            print("\n[SUCCESS] Pen is now streaming over its SoftAP.")
            print()
            print("  Next steps:")
            print(f"    1. Disconnect this PC from its current WiFi network.")
            print(f"    2. Connect to:  SSID '{AP_SSID}'  /  Password '{AP_PASS}'")
            print(f"    3. Run the decoder:   ./build/decoder wifi")
            print(f"    4. Or the TUI:        ./build/tui wifi")
            print()
            print("  The pen broadcasts to 192.168.4.255:5005.")
            print("  Your laptop will get 192.168.4.2 automatically.")
        else:
            print("\n[ERROR] The pen did not confirm WiFi mode.")
            print("        Make sure the firmware was built and flashed from this branch.")

    except serial.SerialException as e:
        print(f"\n[ERROR] Serial connection failed: {e}")
        print("Hint: Close the TUI or decoder first — they hold the USB port.")
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")


if __name__ == "__main__":
    main()
