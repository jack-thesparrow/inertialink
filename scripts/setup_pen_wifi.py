import socket
import getpass
import serial
import serial.tools.list_ports
import time

def get_local_ip():
    try:
        # Create a dummy socket connection to determine the default outgoing route.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

def find_esp32_port():
    print("Scanning for ESP32 USB ports...")
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        desc = str(port.description).lower()
        if 'cp210' in desc or 'ch340' in desc or 'silicon' in desc or 'wch' in desc:
            return port.device
            
    # Fallback to the first available ttyUSB/ttyACM
    for port in ports:
        if 'ttyUSB' in port.device or 'ttyACM' in port.device:
            return port.device

    return None

def main():
    print("=== InertiaLink Smart Pen WiFi Setup ===\n")
    
    port = find_esp32_port()
    if not port:
        print("ERROR: Could not find the ESP32 USB port. Please plug the pen directly into this PC.")
        return
        
    print(f"Detected Pen on: {port}\n")
    
    ssid = input("Enter your WiFi network name (SSID): ").strip()
    if not ssid:
        print("Setup canceled. Network name cannot be empty.")
        return
        
    password = getpass.getpass("Enter your WiFi password (input will be hidden): ")
    
    local_ip = get_local_ip()
    ip_input = input(f"Enter the Desktop IP address where the decoder will run [{local_ip}]: ").strip()
    target_ip = ip_input if ip_input else local_ip
    
    command = f"MODE:WIFI|{ssid}|{password}|{target_ip}\n"
    
    print("\n[Sys] Restarting ESP32 over USB to accept configuration...")
    try:
        # Open port and assert DTR/RTS briefly to trigger auto-reset
        ser = serial.Serial(port, 115200, timeout=0.2)
        ser.setDTR(False)
        ser.setRTS(False)
        
        # Wait for the unit to completely finish boot sequence and announce Ready
        print("[Sys] Waiting for ESP32 to finish booting (could take up to 10s if deleting an old WiFi)...")
        ready = False
        start_wait = time.time()
        while time.time() - start_wait < 15.0:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"  [Boot] {line}")
            if "[ESP] Ready" in line:
                ready = True
                break
                
        if not ready:
            print("\n[WARNING] Never saw the 'Ready' signal! Writing anyway, but this may fail.")
            
        time.sleep(0.5) # Give loop() a tiny margin to spin up
        
        ser.reset_input_buffer()
        print(f"\n[Sys] Sending network binding for '{ssid}' -> {target_ip} ...")
        
        # Write configuration to the ESP32 parser
        ser.write(command.encode("utf-8"))
        ser.flush()
        
        print("[Sys] Waiting for connection confirmation...")
        success = False
        start_wait = time.time()
        
        # Wait up to 15s to hear back if it connected successfully to the new network
        while time.time() - start_wait < 15.0:
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
            print("\n[SUCCESS] Credentials securely flashed into NVS memory!")
            print("    You can now unplug the pen from this PC.")
            print("    When plugged into a portable power bank, it will auto-connect to WiFi.")
        else:
            print("\n[ERROR] The ESP32 failed to connect to that WiFi network.")
            print("    Please double check your 2.4GHz network SSID and password.")
        
    except serial.SerialException as e:
        print(f"\n[ERROR] Serial connection failed: {e}")
        print("Hint: Check if the TUI or Decoder is currently blocking the port! Close them first.")
    except Exception as e:
        print(f"\n[ERROR] Installation failed: {e}")

if __name__ == "__main__":
    main()
