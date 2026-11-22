# Author: Manan Gupta <mnn@yogins.com>
"""Write mumax3 scripts for arrays of independent thermal nanomagnet disks.

One mumax3 process costs about the same per time step for a 10x10 and a
512x512 grid, so every script simulates an array of identical disks, one
region per disk, and tables the disk-averaged magnetization of each one.
mumax3 allows 255 regions, so an array holds at most 255 disks.
"""

import numpy as np

CELL = 2e-9         # in-plane cell size (m)
THICK = 1e-9        # film thickness, one cell (m)


def disk_array_script(nx=16, ny=15, pitch=32, diam=20e-9, ku=6.0e5, ms=1.0e6,
                      aex=15e-12, alpha=0.1, dt=50e-15, save=10e-12, seed=1,
                      p_up=0.5, comps="z", schedule=((0.0, 100e-9),),
                      field_axis="z", stt_j=0.0, stt_pol=0.5, energies=False, cell=CELL):
    """Return the text of a mumax3 script for an nx x ny disk array.

    schedule is a list of (bias, duration) segments run one after the other;
    bias is a field in tesla along field_axis. Each disk starts along +z with
    probability p_up and along -z otherwise (random per disk, from seed).
    stt_j adds a Slonczewski current density (A/m^2) through a fixed layer
    along +z, applied for the whole run; a segment written (bias, duration, j)
    sets the current density to j from that segment on, for current steps.
    pitch is in cells of size cell (m), so a finer mesh needs a larger pitch
    for the same spacing.
    """
    if nx * ny > 255:
        raise ValueError("mumax3 supports at most 255 regions")
    p = pitch * cell
    lines = [
        f"SetGridSize({nx * pitch}, {ny * pitch}, 1)",
        f"SetCellSize({cell}, {cell}, {THICK})",
        f"SetGeom(Circle({diam}).Repeat({p}, {p}, 0).Transl({p / 2}, {p / 2}, 0))",
        f"Msat = {ms}",
        f"Aex = {aex}",
        f"alpha = {alpha}",
        f"Ku1 = {ku}",
        "anisU = vector(0, 0, 1)",
    ]
    if stt_j or any(len(seg) > 2 for seg in schedule):
        lines += [f"J = vector(0, 0, {stt_j})", f"Pol = {stt_pol}", "Lambda = 1",
                  "EpsilonPrime = 0", "FixedLayer = vector(0, 0, 1)"]
    rng = np.random.default_rng(seed)
    k = 0
    for i in range(nx):
        for j in range(ny):
            k += 1
            # Centres sit on cell corners so every disk has the same shape (80 cells at 2 nm).
            x = (i - (nx - 1) / 2) * p
            y = (j - (ny - 1) / 2) * p
            up = 1 if rng.random() < p_up else -1
            lines.append(f"DefRegion({k}, Circle({diam}).Transl({x}, {y}, 0))")
            lines.append(f"m.SetRegion({k}, Uniform(0, 0, {up}))")
            if comps == "z":
                lines.append(f"TableAdd(m.Comp(2).Region({k}))")
            else:
                lines.append(f"TableAdd(m.Region({k}))")
    if energies:
        lines += ["TableAdd(E_demag)", "TableAdd(E_anis)", "TableAdd(E_exch)"]
    axis = {"x": "vector({b}, 0, 0)", "z": "vector(0, 0, {b})"}[field_axis]
    lines += ["Temp = 300", f"ThermSeed({seed})", "SetSolver(2)", f"FixDt = {dt}",
              f"TableAutoSave({save})"]
    for bias, duration, *current in schedule:
        if current:
            lines.append(f"J = vector(0, 0, {current[0]:.6g})")
        lines.append("B_ext = " + axis.format(b=f"{bias:.6g}"))
        lines.append(f"Run({duration:.6g})")
    return "\n".join(lines) + "\n"


def square_wave(b_low, b_high, half, periods):
    """Schedule alternating b_low and b_high for `periods` full periods."""
    return [(b, half) for _ in range(periods) for b in (b_low, b_high)]
