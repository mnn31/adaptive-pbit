# Author: Manan Gupta <mnn@yogins.com>
"""Write every mumax3 script of the device study plus a job list.

Usage:
    python make_runs.py OUTDIR                     # main study at 50 fs, with checks
    python make_runs.py OUTDIR --stt J1 J2 ...     # spin-torque bias runs (A/m^2)
    python make_runs.py OUTDIR --sets eq step --dt 25e-15 --seed-offset 10000
    python make_runs.py OUTDIR --extra             # added seeds and checks at 25 fs
    python make_runs.py OUTDIR --controls          # controls repeated at 25 fs
    python make_runs.py OUTDIR --stt-step J        # spin-torque step response at 25 fs
    python make_runs.py OUTDIR --mesh-cal KU ...   # 1 nm cells: short zero-field runs
    python make_runs.py OUTDIR --mesh KU           # 1 nm cells: the mesh check at KU

The main study writes the bias points, checks, steps, easy-plane control and a
short spin-torque probe; --stt writes only the spin-torque runs at currents
chosen from the probe. At 50 fs the correlation times at +18 and +36 mT came
out shorter than at 25 fs (the mirrored fields, with fewer seeds, show no
difference), so the equilibrium ("eq") and step-response ("step") sets
were rerun at 25 fs with every seed shifted by 10000 (the third line above);
that rerun is the production set. --sets keeps only the named sets of the main
study, --dt replaces their time step and --seed-offset shifts their seeds.
--extra writes the runs added after the rerun: more seeds where statistics
were thin, and an easy-plane step check. --controls writes the 25 fs repeats
of the controls first run at 50 fs under bias (spin torque, easy plane in a
field), with more seeds of the damping check and of the 12.5 fs zero-field
point. --stt-step writes equilibrium runs at current density J and a
square wave between -J and +J, the current-driven counterpart of the field
steps. --mesh-cal and --mesh repeat part of the study with 1 nm cells (see
mesh_jobs). Each job gets OUTDIR/<name>.mx3 and OUTDIR/<name>.json (its
parameters). jobs.txt lists the scripts longest first, so the queue packs
well.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from mx3gen import disk_array_script, square_wave

NS = 1e-9
MT = 1e-3
SLOPE_PILOT = 0.056      # I per mT, from a pilot run; only used to pick initial states
STT_CURRENTS = (1.4e10, 2.4e10)   # A/m^2, chosen from the 50 fs probe (I near 1 and 2)
BURN = 20 * NS           # discarded at the start of every equilibrium run


def p_init(b_mt):
    """Initial up-probability near the expected equilibrium, so burn-in is short."""
    return 0.5 * (1 + np.tanh(SLOPE_PILOT * b_mt))


def stt_jobs(currents, length=150, seeds=2, set_name="stt", seed=5000):
    """Spin-torque bias runs: Slonczewski current through a +z fixed layer, no field."""
    for j in currents:
        for k in range(seeds):
            seed += 1
            yield (f"{set_name}_j{j:.2e}_s{k}".replace("+", ""), set_name,
                   {"nx": 8, "ny": 8, "seed": seed, "stt_j": j, "schedule": [(0.0, BURN + length * NS)]},
                   {"stt_j": j, "bz_mT": 0, "burn_ns": BURN / NS})


def jobs():
    """Yield (name, set, script kwargs, extra metadata) for the whole study."""
    yield from stt_jobs((1e10, 3e10), length=60, seeds=1, set_name="sttprobe", seed=4000)
    seed = 1000
    # Device P equilibrium runs: field in mT -> (seeds, run length after burn-in, ns).
    plan = {0: (4, 200), 9: (4, 150), -9: (2, 150), 18: (4, 150), -18: (2, 150),
            27: (4, 150), -27: (2, 150), 36: (4, 150), -36: (2, 150),
            54: (4, 150), 72: (4, 100), 108: (4, 50)}
    for b, (n, length) in plan.items():
        for k in range(n):
            seed += 1
            yield (f"eq_b{b:+d}_s{k}", "eq",
                   {"nx": 8, "ny": 8, "seed": seed, "p_up": p_init(b),
                        "schedule": [(b * MT, BURN + length * NS)]},
                   {"bz_mT": b, "burn_ns": BURN / NS})
    # Time-step check: dt = 25 fs at zero bias and 18 mT.
    for b in (0, 18):
        for k in range(2):
            seed += 1
            yield (f"dt25_b{b:+d}_s{k}", "dt25",
                   {"nx": 8, "ny": 8, "seed": seed, "p_up": p_init(b), "dt": 25e-15,
                        "schedule": [(b * MT, BURN + 150 * NS)]},
                   {"bz_mT": b, "burn_ns": BURN / NS})
    # Damping check, alpha = 0.03 (dt = 25 fs: 50 fs overheats at low damping).
    for k in range(2):
        seed += 1
        yield (f"alpha03_b+0_s{k}", "alpha03",
               {"nx": 8, "ny": 8, "seed": seed, "alpha": 0.03, "dt": 25e-15,
                    "schedule": [(0.0, BURN + 200 * NS)]},
               {"bz_mT": 0, "burn_ns": BURN / NS})
    # Crosstalk control: 4x4 array at twice the pitch (128 nm).
    seed += 1
    yield ("pitch128_b+0_s0", "pitch128",
           {"nx": 4, "ny": 4, "pitch": 64, "seed": seed, "schedule": [(0.0, BURN + 200 * NS)]},
           {"bz_mT": 0, "burn_ns": BURN / NS})
    # Step response: square-wave field, (low mT, high mT, half period ns), 4 periods.
    for lo, hi, half in ((-18, 18, 50), (-36, 36, 30), (-72, 72, 15), (-108, 108, 10),
                         (0, 18, 60)):
        for k in range(2):
            seed += 1
            yield (f"step_{lo:+d}_{hi:+d}_s{k}", "step",
                   {"nx": 8, "ny": 8, "seed": seed, "p_up": p_init(lo),
                        "schedule": [(lo * MT, BURN)] + square_wave(hi * MT, lo * MT, half * NS, 4)},
                   {"low_mT": lo, "high_mT": hi, "half_ns": half, "periods": 4, "burn_ns": BURN / NS})
    # Device E: easy-plane control (Ku = 0), all three components, in-plane bias.
    for b, length in ((0, 30), (2, 20), (5, 20), (10, 20), (20, 20), (40, 20)):
        for k in range(2):
            seed += 1
            yield (f"ep_bx{b:+d}_s{k}", "easyplane",
                   {"nx": 8, "ny": 8, "seed": seed, "ku": 0.0, "comps": "xyz", "save": 5e-12,
                        "field_axis": "x", "schedule": [(b * MT, 2 * NS + length * NS)]},
                   {"bx_mT": b, "burn_ns": 2.0})
    # The I = 1 point carries the weak-bias comparisons, so it gets more seeds
    # (appended here so the seeds of the runs above do not change).
    for b, first, n in ((18, 4, 4), (-18, 2, 2)):
        for k in range(first, first + n):
            seed += 1
            yield (f"eq_b{b:+d}_s{k}", "eq",
                   {"nx": 8, "ny": 8, "seed": seed, "p_up": p_init(b),
                        "schedule": [(b * MT, BURN + 150 * NS)]},
                   {"bz_mT": b, "burn_ns": BURN / NS})
    for k in (2, 3):
        seed += 1
        yield (f"dt25_b+18_s{k}", "dt25",
               {"nx": 8, "ny": 8, "seed": seed, "p_up": p_init(18), "dt": 25e-15,
                    "schedule": [(18 * MT, BURN + 150 * NS)]},
               {"bz_mT": 18, "burn_ns": BURN / NS})
    # Weak-bias steps with an 80 ns half period, longer than a settle of 4 tau_c(0),
    # so the segment tail is a usable asymptote: (low mT, high mT, seeds), 3 periods.
    for lo, hi, n in ((-18, 18, 6), (0, 18, 8)):
        for k in range(n):
            seed += 1
            yield (f"step_{lo:+d}_{hi:+d}_h80_s{k}", "step",
                   {"nx": 8, "ny": 8, "seed": seed, "p_up": p_init(lo),
                        "schedule": [(lo * MT, BURN)] + square_wave(hi * MT, lo * MT, 80 * NS, 3)},
                   {"low_mT": lo, "high_mT": hi, "half_ns": 80, "periods": 3, "burn_ns": BURN / NS})
    # Time-step check with enough seeds to resolve a 10% change of tau_c under bias.
    for b, ks in ((18, range(4, 12)), (0, range(2, 6)), (36, range(4))):
        for k in ks:
            seed += 1
            yield (f"dt25_b{b:+d}_s{k}", "dt25",
                   {"nx": 8, "ny": 8, "seed": seed, "p_up": p_init(b), "dt": 25e-15,
                        "schedule": [(b * MT, BURN + 150 * NS)]},
                   {"bz_mT": b, "burn_ns": BURN / NS})
    # The 25 fs runs differ from 50 fs under bias, so 12.5 fs is run as well.
    for b, n in ((0, 3), (18, 6), (36, 7)):
        for k in range(n):
            seed += 1
            yield (f"dt12_b{b:+d}_s{k}", "dt12",
                   {"nx": 8, "ny": 8, "seed": seed, "p_up": p_init(b), "dt": 12.5e-15,
                        "schedule": [(b * MT, BURN + 120 * NS)]},
                   {"bz_mT": b, "burn_ns": BURN / NS})


def rerun(todo, sets, dt=None, seed_offset=0):
    """Keep the jobs of the named sets, with a new time step and shifted seeds."""
    for name, group, kw, meta in todo:
        if sets and group not in sets:
            continue
        kw = dict(kw, dt=dt or kw.get("dt", 50e-15), seed=kw["seed"] + seed_offset)
        yield name, group, kw, meta


def extra_jobs():
    """Runs added at 25 fs after the production rerun (seeds from 20001, distinct from all others)."""
    seed = 20000
    # More seeds where the production set had few seeds or few minority visits.
    for b, first, n in ((-9, 2, 2), (-27, 2, 2), (-36, 2, 2), (54, 4, 4), (72, 4, 4)):
        for k in range(first, first + n):
            seed += 1
            yield (f"eq_b{b:+d}_s{k}", "eq",
                   {"nx": 8, "ny": 8, "seed": seed, "p_up": p_init(b), "dt": 25e-15,
                        "schedule": [(b * MT, BURN + 150 * NS)]},
                   {"bz_mT": b, "burn_ns": BURN / NS})
    # Easy-plane control at 25 fs, to check that its conclusions do not depend on the step.
    for k in range(2):
        seed += 1
        yield (f"ep25_bx+0_s{k}", "easyplane25",
               {"nx": 8, "ny": 8, "seed": seed, "ku": 0.0, "comps": "xyz", "save": 5e-12,
                    "dt": 25e-15, "field_axis": "x", "schedule": [(0.0, 2 * NS + 30 * NS)]},
               {"bx_mT": 0, "burn_ns": 2.0})


def control_jobs():
    """25 fs repeats of the controls, and more seeds of two checks (seeds from 30001)."""
    # Spin-torque bias: the currents of the 50 fs runs, four seeds each.
    for name, group, kw, meta in stt_jobs(STT_CURRENTS, seeds=4, seed=30000):
        yield name, group, dict(kw, dt=25e-15), meta
    seed = 30100
    # Easy plane in an in-plane field (the zero-field point is in extra_jobs).
    for b in (2, 5, 10, 20, 40):
        for k in range(2):
            seed += 1
            yield (f"ep25_bx{b:+d}_s{k}", "easyplane25",
                   {"nx": 8, "ny": 8, "seed": seed, "ku": 0.0, "comps": "xyz", "save": 5e-12,
                        "dt": 25e-15, "field_axis": "x", "schedule": [(b * MT, 2 * NS + 20 * NS)]},
                   {"bx_mT": b, "burn_ns": 2.0})
    # Damping check: two more seeds, so its error rests on four.
    for k in (2, 3):
        seed += 1
        yield (f"alpha03_b+0_s{k}", "alpha03",
               {"nx": 8, "ny": 8, "seed": seed, "alpha": 0.03, "dt": 25e-15,
                    "schedule": [(0.0, BURN + 200 * NS)]},
               {"bz_mT": 0, "burn_ns": BURN / NS})
    # 12.5 fs at zero field: tau_c(0) from three seeds was too uncertain a reference
    # for the 12.5 fs cosh-law ratios.
    for k in range(3, 8):
        seed += 1
        yield (f"dt12_b+0_s{k}", "dt12",
               {"nx": 8, "ny": 8, "seed": seed, "p_up": p_init(0), "dt": 12.5e-15,
                    "schedule": [(0.0, BURN + 120 * NS)]},
               {"bz_mT": 0, "burn_ns": BURN / NS})


def stt_step_jobs(j, seeds=2, half=15, periods=4):
    """Current steps at 25 fs: equilibrium runs at +j, then a square wave between -j and +j."""
    for name, group, kw, meta in stt_jobs((j,), seeds=4, seed=40000):
        yield name, group, dict(kw, dt=25e-15), meta
    for k in range(seeds):
        wave = [(0.0, half * NS, x) for _ in range(periods) for x in (j, -j)]
        yield (f"sttstep_j{j:.2e}_s{k}".replace("+", ""), "sttstep",
               {"nx": 8, "ny": 8, "seed": 40100 + k, "p_up": 0.0, "dt": 25e-15,
                "schedule": [(0.0, BURN, -j)] + wave},
               {"low_j": -j, "high_j": j, "half_ns": half, "periods": periods,
                "burn_ns": BURN / NS, "bz_mT": 0})


# Mesh check: 1 nm cells need a 6.25 fs step for the same exchange-mode
# accuracy as 25 fs at 2 nm (the stiffest mode scales as 1 / cell^2). The grid
# cost then scales with the array area, so the pitch is 48 nm (64 nm in the
# main study) and the neighbour correlation of the reads is recorded.
MESH = {"nx": 8, "ny": 8, "cell": 1e-9, "pitch": 48, "dt": 6.25e-15}
MESH_BURN = 2 * NS       # disks start in a well, so only in-well motion has to relax


def mesh_cal_jobs(kus, length=28):
    """One short zero-field run per anisotropy constant, to find Ku for the 2 nm barrier."""
    for n, ku in enumerate(kus):
        yield (f"meshcal_ku{ku / 1e3:.0f}", "mesh",
               dict(MESH, seed=50001 + n, ku=ku, schedule=[(0.0, MESH_BURN + length * NS)]),
               {"bz_mT": 0, "burn_ns": MESH_BURN / NS})


def mesh_jobs(ku):
    """Mesh check at one Ku: zero field, I near 1, and steps between -72 and +72 mT."""
    seed = 51000
    for b, n, length in ((0, 3, 50), (18, 1, 50)):
        for k in range(n):
            seed += 1
            yield (f"mesh_b{b:+d}_s{k}", "mesh",
                   dict(MESH, seed=seed, ku=ku, p_up=p_init(b),
                        schedule=[(b * MT, MESH_BURN + length * NS)]),
                   {"bz_mT": b, "burn_ns": MESH_BURN / NS})
    for k in range(2):
        seed += 1
        yield (f"meshstep_-72_+72_s{k}", "meshstep",
               dict(MESH, seed=seed, ku=ku, p_up=p_init(-72),
                    schedule=[(-72 * MT, MESH_BURN)] + square_wave(72 * MT, -72 * MT, 10 * NS, 3)),
               {"low_mT": -72, "high_mT": 72, "half_ns": 10, "periods": 3,
                "burn_ns": MESH_BURN / NS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--stt", type=float, nargs="+", help="spin-torque current densities (A/m^2)")
    ap.add_argument("--sets", nargs="+", help="keep only these sets of the main study")
    ap.add_argument("--dt", type=float, help="time step (s) for the kept jobs")
    ap.add_argument("--seed-offset", type=int, default=0, help="added to every kept seed")
    ap.add_argument("--extra", action="store_true", help="the runs added at 25 fs")
    ap.add_argument("--controls", action="store_true", help="the controls repeated at 25 fs")
    ap.add_argument("--stt-step", type=float, help="current density (A/m^2) of the current steps")
    ap.add_argument("--mesh-cal", type=float, nargs="+", help="Ku values (J/m^3) of the 1 nm scan")
    ap.add_argument("--mesh", type=float, help="Ku (J/m^3) of the 1 nm mesh check")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if a.stt:
        todo = stt_jobs(a.stt)
    elif a.extra:
        todo = extra_jobs()
    elif a.controls:
        todo = control_jobs()
    elif a.stt_step:
        todo = stt_step_jobs(a.stt_step)
    elif a.mesh_cal:
        todo = mesh_cal_jobs(a.mesh_cal)
    elif a.mesh:
        todo = mesh_jobs(a.mesh)
    else:
        todo = jobs()
    if a.sets or a.dt or a.seed_offset:
        todo = rerun(todo, a.sets, a.dt, a.seed_offset)
    rows = []
    for name, group, kw, meta in todo:
        (out / f"{name}.mx3").write_text(disk_array_script(**kw))
        sim_ns = sum(seg[1] for seg in kw["schedule"]) / NS
        dt = kw.get("dt", 50e-15)
        meta = dict(meta, name=name, set=group, sim_ns=sim_ns,
                    **{k: v for k, v in kw.items() if k != "schedule"},
                    schedule=[list(seg) for seg in kw["schedule"]])
        meta.setdefault("dt", dt)
        (out / f"{name}.json").write_text(json.dumps(meta, indent=1))
        rows.append((sim_ns * 50e-15 / dt, name))
    rows.sort(reverse=True)
    (out / "jobs.txt").write_text("".join(f"{n}.mx3\n" for _, n in rows))
    print(len(rows), "jobs,", round(sum(c for c, _ in rows)), "ns-at-50fs in total")


if __name__ == "__main__":
    main()
