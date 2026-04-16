#!/usr/bin/env python3
"""
InertiaLink Calibration Script
===============================
Trace a known 5 cm square on paper to calibrate the visualizer's axis
mapping, scaling, and sign conventions.

Usage:
    python3 scripts/calibrate.py [port]

    port  — serial port (default: /dev/ttyUSB0)

Procedure:
    1. Place the pen at the TOP-LEFT corner of the square.
    2. Press Enter, then trace the TOP edge → RIGHT     (5 cm rightward)
    3. Press Enter, then trace the RIGHT edge → DOWN     (5 cm downward)
    4. Press Enter, then trace the BOTTOM edge → LEFT    (5 cm leftward)
    5. Press Enter, then trace the LEFT edge → UP        (5 cm upward)
    6. Press Enter to finish.

The script records raw 6-DOF data for each edge, integrates gyro and
double-integrates accel, and computes the calibration constants.
"""

import serial
import sys
import time
import math

def median(vals):
    """Simple median without numpy."""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

# ── Configuration ───────────────────────────────────────────────────────────
PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
BAUD = 115200
DT = 0.01  # 100 Hz
SQUARE_SIDE_MM = 50.0  # 5 cm

# Current visualizer defaults (from io.hpp / viz.cpp)
TILT_ANGLE_DEG = 35.0
LEVER_ARM_MM = 150.0
MM_TO_GL = 1.0 / 200.0
POS_TO_GL = 50.0
GYRO_WEIGHT = 0.7
ACCEL_WEIGHT = 0.3
G_MPS2 = 9.80665

DEG2RAD = math.pi / 180.0
TILT_RAD = TILT_ANGLE_DEG * DEG2RAD
COS_T = math.cos(TILT_RAD)
SIN_T = math.sin(TILT_RAD)


def parse_imu_line(line: str):
    """Parse 'ax,ay,az,gx,gy,gz' → dict or None."""
    try:
        parts = line.strip().split(",")
        if len(parts) == 6:
            vals = [float(x) for x in parts]
            return {
                "ax": vals[0], "ay": vals[1], "az": vals[2],
                "gx": vals[3], "gy": vals[4], "gz": vals[5],
            }
    except (ValueError, IndexError):
        pass
    return None


def record_edge(ser, edge_name, direction_hint):
    """Record one edge of the square. Returns list of IMU dicts."""
    input(f"\n  ▶  Place pen at start of {edge_name} edge.\n"
          f"     You will trace {direction_hint}.\n"
          f"     Press ENTER to START recording, then trace the edge slowly...")

    # Flush stale data
    ser.reset_input_buffer()
    time.sleep(0.1)

    samples = []
    print(f"     ⏺  RECORDING — trace the {edge_name} edge now...")
    print(f"     Press ENTER when you reach the end of this edge.")

    import threading
    stop_flag = threading.Event()

    def wait_for_enter():
        input()
        stop_flag.set()

    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()

    while not stop_flag.is_set():
        raw = ser.readline()
        if raw:
            line = raw.decode("utf-8", errors="ignore").strip()
            imu = parse_imu_line(line)
            if imu:
                samples.append(imu)

    print(f"     ✓  Captured {len(samples)} samples for {edge_name} edge "
          f"({len(samples) * DT:.1f}s)")
    return samples


def integrate_edge(samples):
    """
    Integrate gyro → angle → lever-arm position AND
    double-integrate accel → velocity → position for one edge.

    Returns dict with final integrated positions from both methods.
    """
    # Tilt-compensated gyro integration
    int_pitch = 0.0
    int_yaw = 0.0
    int_roll = 0.0

    # Gravity estimation (complementary filter)
    grav_x, grav_y, grav_z = 0.0, 0.0, 1.0
    vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
    pos_x, pos_y, pos_z = 0.0, 0.0, 0.0

    VEL_DECAY = 0.92
    ACCEL_GATE = 0.15
    GRAV_ALPHA = 0.98

    # Raw gyro accumulations (no tilt compensation)
    raw_gx_sum = 0.0
    raw_gy_sum = 0.0
    raw_gz_sum = 0.0

    for s in samples:
        # Tilt-compensated gyro
        gx_c = s["gx"] * COS_T + s["gz"] * SIN_T
        gz_c = -s["gx"] * SIN_T + s["gz"] * COS_T
        gy_c = s["gy"]

        int_pitch += gx_c * DT * DEG2RAD
        int_yaw += gy_c * DT * DEG2RAD
        int_roll += gz_c * DT * DEG2RAD

        # Raw gyro integration (for comparison)
        raw_gx_sum += s["gx"] * DT * DEG2RAD
        raw_gy_sum += s["gy"] * DT * DEG2RAD
        raw_gz_sum += s["gz"] * DT * DEG2RAD

        # Tilt-compensated accel
        ax_c = s["ax"] * COS_T + s["az"] * SIN_T
        az_c = -s["ax"] * SIN_T + s["az"] * COS_T
        ay_c = s["ay"]

        # Gravity filter
        grav_x = GRAV_ALPHA * grav_x + (1 - GRAV_ALPHA) * ax_c
        grav_y = GRAV_ALPHA * grav_y + (1 - GRAV_ALPHA) * ay_c
        grav_z = GRAV_ALPHA * grav_z + (1 - GRAV_ALPHA) * az_c

        # Dynamic accel
        dyn_ax = (ax_c - grav_x) * G_MPS2
        dyn_ay = (ay_c - grav_y) * G_MPS2
        dyn_az = (az_c - grav_z) * G_MPS2

        dyn_mag = math.sqrt(dyn_ax**2 + dyn_ay**2 + dyn_az**2)
        if dyn_mag > ACCEL_GATE:
            vel_x = vel_x * VEL_DECAY + dyn_ax * DT
            vel_y = vel_y * VEL_DECAY + dyn_ay * DT
            vel_z = vel_z * VEL_DECAY + dyn_az * DT
        else:
            vel_x *= VEL_DECAY * 0.8
            vel_y *= VEL_DECAY * 0.8
            vel_z *= VEL_DECAY * 0.8

        pos_x += vel_x * DT
        pos_y += vel_y * DT
        pos_z += vel_z * DT

    # Gyro lever-arm position (current visualizer formula, with face-up signs)
    gyro_screen_x = int_yaw * LEVER_ARM_MM * MM_TO_GL
    gyro_screen_y = -int_pitch * LEVER_ARM_MM * MM_TO_GL

    # Accel position (current visualizer formula, face-up signs)
    accel_screen_x = pos_y * POS_TO_GL
    accel_screen_y = pos_x * POS_TO_GL

    # Blended
    blend_x = GYRO_WEIGHT * gyro_screen_x + ACCEL_WEIGHT * accel_screen_x
    blend_y = GYRO_WEIGHT * gyro_screen_y + ACCEL_WEIGHT * accel_screen_y

    # Gyro lever-arm in mm (physical estimate)
    gyro_mm_x = int_yaw * LEVER_ARM_MM
    gyro_mm_y = -int_pitch * LEVER_ARM_MM

    return {
        "int_pitch": int_pitch, "int_yaw": int_yaw, "int_roll": int_roll,
        "raw_gx": raw_gx_sum, "raw_gy": raw_gy_sum, "raw_gz": raw_gz_sum,
        "gyro_screen_x": gyro_screen_x, "gyro_screen_y": gyro_screen_y,
        "gyro_mm_x": gyro_mm_x, "gyro_mm_y": gyro_mm_y,
        "accel_screen_x": accel_screen_x, "accel_screen_y": accel_screen_y,
        "accel_pos": (pos_x, pos_y, pos_z),
        "blend_x": blend_x, "blend_y": blend_y,
        "n_samples": len(samples),
    }


def main():
    print("=" * 64)
    print("  InertiaLink Calibration")
    print("=" * 64)
    print(f"  Port:          {PORT}")
    print(f"  Square side:   {SQUARE_SIDE_MM} mm")
    print(f"  Tilt angle:    {TILT_ANGLE_DEG}°")
    print(f"  Lever arm:     {LEVER_ARM_MM} mm")
    print()

    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)  # wait for ESP32 DTR reset
    ser.reset_input_buffer()

    # Verify connection
    print("  Checking IMU data stream...")
    got_data = False
    for _ in range(50):
        raw = ser.readline()
        if raw:
            line = raw.decode("utf-8", errors="ignore").strip()
            if parse_imu_line(line):
                got_data = True
                break
    if not got_data:
        print("  ✗ No IMU data received. Check connection.")
        ser.close()
        return

    print("  ✓ IMU data streaming.\n")
    print("  Instructions:")
    print("  Draw a 5 cm × 5 cm square on paper.")
    print("  You will trace each edge one at a time:")
    print("    Edge 1: TOP    — trace LEFT → RIGHT  (rightward)")
    print("    Edge 2: RIGHT  — trace TOP → BOTTOM  (downward)")
    print("    Edge 3: BOTTOM — trace RIGHT → LEFT  (leftward)")
    print("    Edge 4: LEFT   — trace BOTTOM → TOP  (upward)")

    edges = [
        ("TOP",    "LEFT → RIGHT (rightward,  +X on screen)"),
        ("RIGHT",  "TOP → BOTTOM (downward,   -Y on screen)"),
        ("BOTTOM", "RIGHT → LEFT (leftward,   -X on screen)"),
        ("LEFT",   "BOTTOM → TOP (upward,     +Y on screen)"),
    ]

    # Expected screen displacements for each edge (in "ideal" units)
    # RIGHT: dx=+50mm, dy=0
    # DOWN:  dx=0,     dy=-50mm
    # LEFT:  dx=-50mm, dy=0
    # UP:    dx=0,     dy=+50mm
    expected_mm = [
        (SQUARE_SIDE_MM, 0.0),
        (0.0, -SQUARE_SIDE_MM),
        (-SQUARE_SIDE_MM, 0.0),
        (0.0, SQUARE_SIDE_MM),
    ]

    results = []
    for (name, hint), (ex, ey) in zip(edges, expected_mm):
        samples = record_edge(ser, name, hint)
        if len(samples) < 10:
            print(f"     ⚠  Too few samples for {name} edge, skipping.")
            results.append(None)
            continue
        r = integrate_edge(samples)
        r["expected_dx_mm"] = ex
        r["expected_dy_mm"] = ey
        results.append(r)

    ser.close()

    # ── Analysis ────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  CALIBRATION RESULTS")
    print("=" * 64)

    print("\n  ── Raw Gyro Integration (radians) per edge ──")
    print(f"  {'Edge':<8} {'pitch(gx)':>10} {'yaw(gy)':>10} {'roll(gz)':>10}")
    for (name, _), r in zip(edges, results):
        if r:
            print(f"  {name:<8} {r['int_pitch']:>+10.4f} {r['int_yaw']:>+10.4f} "
                  f"{r['int_roll']:>+10.4f}")

    print("\n  ── Gyro Lever-Arm Position (mm) per edge ──")
    print(f"  {'Edge':<8} {'screen_X mm':>12} {'screen_Y mm':>12} "
          f"{'expected_X':>11} {'expected_Y':>11}")
    for (name, _), r in zip(edges, results):
        if r:
            print(f"  {name:<8} {r['gyro_mm_x']:>+12.2f} {r['gyro_mm_y']:>+12.2f} "
                  f"{r['expected_dx_mm']:>+11.1f} {r['expected_dy_mm']:>+11.1f}")

    print("\n  ── Accel Double-Integration (m) per edge ──")
    print(f"  {'Edge':<8} {'pos_X(m)':>10} {'pos_Y(m)':>10} {'pos_Z(m)':>10} "
          f"{'screen_X':>10} {'screen_Y':>10}")
    for (name, _), r in zip(edges, results):
        if r:
            px, py, pz = r["accel_pos"]
            print(f"  {name:<8} {px:>+10.5f} {py:>+10.5f} {pz:>+10.5f} "
                  f"{r['accel_screen_x']:>+10.4f} {r['accel_screen_y']:>+10.4f}")

    # ── Compute recommended parameters ──────────────────────────────────────
    print("\n  ── Sign Analysis ──")
    print("  Checking if axis signs match expected directions...")

    sign_issues = []
    for (name, _), (ex, ey), r in zip(edges, expected_mm, results):
        if not r:
            continue
        gx, gy = r["gyro_mm_x"], r["gyro_mm_y"]

        # Check dominant axis direction
        if abs(ex) > 0:  # horizontal edge
            if (ex > 0 and gx < 0) or (ex < 0 and gx > 0):
                sign_issues.append(f"  ⚠  {name}: gyro_X sign inverted "
                                   f"(got {gx:+.2f} mm, expected {ex:+.1f} mm)")
            else:
                print(f"  ✓  {name}: gyro_X sign correct ({gx:+.2f} mm)")
        if abs(ey) > 0:  # vertical edge
            if (ey > 0 and gy < 0) or (ey < 0 and gy > 0):
                sign_issues.append(f"  ⚠  {name}: gyro_Y sign inverted "
                                   f"(got {gy:+.2f} mm, expected {ey:+.1f} mm)")
            else:
                print(f"  ✓  {name}: gyro_Y sign correct ({gy:+.2f} mm)")

    if sign_issues:
        print("\n  Sign issues detected:")
        for issue in sign_issues:
            print(issue)
    else:
        print("\n  ✓  All axis signs are correct!")

    # ── Lever arm calibration ───────────────────────────────────────────────
    print("\n  ── Lever Arm Calibration ──")
    lever_estimates = []
    for (name, _), (ex, ey), r in zip(edges, expected_mm, results):
        if not r:
            continue
        # For horizontal edges: expected_dx = yaw_rad * leverArm
        # → leverArm = expected_dx / yaw_rad
        if abs(ex) > 0 and abs(r["int_yaw"]) > 0.001:
            est = abs(ex) / abs(r["int_yaw"])
            lever_estimates.append(est)
            print(f"  {name}: lever_arm from yaw = {est:.1f} mm "
                  f"(yaw = {r['int_yaw']:+.4f} rad)")
        # For vertical edges: expected_dy = pitch_rad * leverArm
        if abs(ey) > 0 and abs(r["int_pitch"]) > 0.001:
            est = abs(ey) / abs(r["int_pitch"])
            lever_estimates.append(est)
            print(f"  {name}: lever_arm from pitch = {est:.1f} mm "
                  f"(pitch = {r['int_pitch']:+.4f} rad)")

    if lever_estimates:
        recommended_lever = median(lever_estimates)
        print(f"\n  Current leverArmMm:     {LEVER_ARM_MM:.1f} mm")
        print(f"  Recommended leverArmMm: {recommended_lever:.1f} mm")
    else:
        recommended_lever = LEVER_ARM_MM
        print("  ⚠  Could not estimate lever arm (gyro angles too small)")

    # ── Blended position scale ──────────────────────────────────────────────
    print("\n  ── GL Scale Calibration ──")
    # Compute what GL scale would make 50mm = some nice fraction of viewport
    # The ortho viewport is ±1.5 by default. 50mm should be about 0.5-0.8 GL units.
    target_gl_per_50mm = 0.6  # how much of the viewport 50mm should occupy

    gl_displacements = []
    for (name, _), (ex, ey), r in zip(edges, expected_mm, results):
        if not r:
            continue
        gl_d = math.sqrt(r["blend_x"]**2 + r["blend_y"]**2)
        if gl_d > 0.001:
            gl_displacements.append(gl_d)
            actual_mm = math.sqrt(ex**2 + ey**2)
            print(f"  {name}: {actual_mm:.0f} mm → {gl_d:.4f} GL units")

    if gl_displacements:
        median_gl = median(gl_displacements)
        if median_gl > 0.001:
            scale_factor = target_gl_per_50mm / median_gl
            new_mm_to_gl = MM_TO_GL * scale_factor
            new_pos_to_gl = POS_TO_GL * scale_factor
            print(f"\n  Current  MM_TO_GL:  {MM_TO_GL}")
            print(f"  Current  POS_TO_GL: {POS_TO_GL}")
            print(f"  Recommended MM_TO_GL:  {new_mm_to_gl:.6f}")
            print(f"  Recommended POS_TO_GL: {new_pos_to_gl:.1f}")
        else:
            new_mm_to_gl = MM_TO_GL
            new_pos_to_gl = POS_TO_GL
    else:
        new_mm_to_gl = MM_TO_GL
        new_pos_to_gl = POS_TO_GL

    # ── Cross-axis bleed ───────────────────────────────────────────────────
    print("\n  ── Cross-Axis Bleed ──")
    print("  (When tracing a horizontal edge, how much vertical movement?)")
    for (name, _), (ex, ey), r in zip(edges, expected_mm, results):
        if not r:
            continue
        gx, gy = abs(r["gyro_mm_x"]), abs(r["gyro_mm_y"])
        if abs(ex) > 0:  # horizontal edge → gy should be near 0
            bleed = gy / max(gx, 0.01) * 100
            print(f"  {name} (horiz): Y bleed = {bleed:.1f}% of X displacement")
        if abs(ey) > 0:  # vertical edge → gx should be near 0
            bleed = gx / max(gy, 0.01) * 100
            print(f"  {name} (vert):  X bleed = {bleed:.1f}% of Y displacement")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  RECOMMENDED CONSTANTS FOR viz.cpp / io.hpp")
    print("=" * 64)

    need_flip_x = any("gyro_X sign inverted" in s for s in sign_issues)
    need_flip_y = any("gyro_Y sign inverted" in s for s in sign_issues)

    gyro_x_sign = "-" if need_flip_x else "+"
    gyro_y_sign = "+" if need_flip_y else "-"
    accel_x_sign = "-" if need_flip_x else "+"
    accel_y_sign = "-" if need_flip_y else "+"

    print(f"""
  // io.hpp — Defaults
  static constexpr float leverArmMm = {recommended_lever:.1f}f;

  // viz.cpp — Canvas projection
  constexpr float MM_TO_GL  = {new_mm_to_gl:.6f}f;
  constexpr float POS_TO_GL = {new_pos_to_gl:.1f}f;

  // viz.cpp — Gyro lever arm (line ~478)
  float gyro_x = {gyro_x_sign}intYaw * pen::Defaults::leverArmMm * MM_TO_GL;
  float gyro_y = {gyro_y_sign}intPitch * pen::Defaults::leverArmMm * MM_TO_GL;

  // viz.cpp — Accel position (line ~484)
  float accel_x = {accel_x_sign}position.y * POS_TO_GL;
  float accel_y = {accel_y_sign}position.x * POS_TO_GL;
""")

    print("=" * 64)
    print("  Calibration complete! Apply the values above to your code.")
    print("=" * 64)


if __name__ == "__main__":
    main()
