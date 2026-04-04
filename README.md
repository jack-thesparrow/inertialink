# Inertialink

## Inertialink: Motion-Sensor Based Smart Pen for Digital Handwriting Capture

- To initialize the program before installation run the `bootstrap.sh` script.
  If you are running this script for the first time then you need to either
  reboot or logout and log back in.
- Configure `cmake` using the command:

  ```bash
  cmake -B build/ -G Ninja
  ```
- Then build the package using the command:
  ```bash
  cmake --build build/
  ```

_Now your build files will be available in the `bin/` folder._

---

## Simulation without hardware

Open a new separate terminal session. Add a virtual `tty` process to run in the
background using:

```bash
socat PTY,link=/tmp/vtty_esp32,raw,echo=0 PTY,link=/tmp/vtty_laptop,raw,echo=0 &
```

Now in a python virtual environment run the `mock_esp32.py`:

```bash
python3 scripts/mock_esp32.py
```

This creates synthetic strokes of sine waves emulating the pen stroke. Visualize
them using `bin/visualizer`.
