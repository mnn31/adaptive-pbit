# Author: Manan Gupta <mnn@yogins.com>
"""Reduce the two checks run after the main study: current steps and a finer mesh.

Usage: python analyze_checks.py RUNDIR [RUNDIR ...] RECORDS

RECORDS is results/micromagnetics/, already written by analyze.py; its
records are read and not changed. Two records are added.

spin_torque_step.json: the response of the read mean to a reversal of the
spin-torque current (no field), the current-driven counterpart of the field
steps in step_response.json. The equilibrium runs at the step current give
the stationary read mean and I; the run at -J is the mirror image of the
run at +J. The step time is compared with the cosh-law time tau_c(0) / cosh I.

mesh_check.json: runs with 1 nm cells (6.25 fs step, 48 nm pitch), grouped by
anisotropy constant and set beside the 2 nm production values. Thermal
fluctuations of the cells lower the barrier, so Ku is a mesh-level input: the
same Ku gives a different barrier on the finer mesh, and the comparison of
dynamics is made at the Ku whose barrier is closest to the 2 nm one. Each Ku
is measured in units of its own tau_c(0). A step to +-72 mT is compared
through tau_step / tau_c(0), which needs no estimate of I, and through the
cosh-law ratio with I taken as 4 x I(18 mT) on both meshes.
"""

import json
import sys
from pathlib import Path

import analyze as an
import mmstats as ms
import numpy as np


# Reads of every half period of square-wave runs by step direction: (table step,
# segments).
def step_segments(members, half_ns):
    segs = {"up": [], "down": []}
    for stem in members:
        meta, dt, _, m = an.load(stem)
        s = ms.sign_reads(m)
        half = round(half_ns / dt)
        for j in range(2 * meta["periods"]):
            seg = s[j * half:(j + 1) * half + 1]
            if len(seg) == half + 1:
                segs["up" if j % 2 == 0 else "down"].append(seg.T)
    return dt, segs


# Step records for the current reversals, built with analyze.step_record.
def current_steps(step_stems, points, ref, rng):
    """Step responses to a current reversal between -J and +J, against the cosh law.

    points are equilibrium points keyed by current density and ref is the
    zero-bias point of the same time step. Where a seed never visits the
    minority state at the step current, I is the line through the origin
    fitted to the currents with a measured logit.
    """
    tau0_pair = ref["tau_efold"]
    tau0 = tau0_pair[0]
    known = [(j, pt["I"][0]) for j, pt in points.items() if pt["I_all_seeds_finite"]]
    line = sum(j * i for j, i in known) / sum(j * j for j, _ in known)
    groups = {}
    for stem in step_stems:
        meta = an.meta_of(stem)
        groups.setdefault((meta["low_j"], meta["high_j"], meta["half_ns"]), []).append(stem)
    out, curves = [], {}
    for (lo, hi, half_ns), members in sorted(groups.items()):
        dt, segs = step_segments(members, half_ns)
        for direction, target in (("up", hi), ("down", lo)):
            sign = 1 if target > 0 else -1
            pt = points[abs(target)]
            s_eq = [sign * (2 * pt["p_up"][0] - 1), 2 * pt["p_up"][1]]
            if pt["I_all_seeds_finite"]:
                i_options = {"measured" if sign == 1 else "mirrored": [sign * pt["I"][0], pt["I"][1]]}
            else:
                i_options = {"line_through_origin": [line * target, np.nan]}
            i_target = next(iter(i_options.values()))[0]
            tau_pred = tau0 / np.cosh(i_target)
            rec, curve = an.step_record(np.concatenate(segs[direction]), dt, s_eq, tau0, tau_pred, rng)
            rec = {"low_j": lo, "high_j": hi, "half_ns": half_ns, "direction": direction,
                   "target_j": target, "I_target": i_target, "I_target_options": i_options,
                   "I_per_j_line": line, "flips_at_target": pt["flips"], "pred_tau_cosh": tau_pred,
                   "tau_step_over_tau0": {k: an.ratio(v, tau0_pair) for k, v in rec["tau_step"].items()},
                   "tau_step_over_cosh": an.step_over_cosh(rec, tau0_pair, i_options)} | rec
            curves[f"j{hi:.2e}_h{half_ns}_{direction}"] = curve
            out.append(rec)
    return out, curves


# Fraction of reads that differ from the one before, over the telegraph value, per r0.
def read_change(rows):
    return {f"{r['r0']:g}": {"changed": [1 - r["measured"]["all"][0], r["measured"]["all"][1]],
                             "telegraph": 1 - r["telegraph_efold"]["all"],
                             "over_telegraph": (1 - r["measured"]["all"][0])
                             / (1 - r["telegraph_efold"]["all"]),
                             "by_construction": r["telegraph_efold"]["by_construction"]}
            for r in rows}


# Steps to +-72 mT: step time over tau_c(0), and over the cosh-law time at I = 4 x I(18
# mT).
def step_rows(recs, tau0_pair, i18):
    rows = []
    for st in recs:
        if abs(st["target_mT"]) != 72:
            continue
        i_line = [float(4 * i18[0] * np.sign(st["target_mT"])), 4 * i18[1]]
        rows.append({"target_mT": st["target_mT"], "half_ns": st["half_ns"], "steps": st["steps"],
                     "s_inf_eq": st["s_inf_eq"], "tau_step_ns": st["tau_step"],
                     "tau_step_over_tau0": {k: an.ratio(v, tau0_pair) for k, v in st["tau_step"].items()},
                     "I_4x18mT": i_line,
                     "tau_step_over_cosh_4x18mT": an.step_over_cosh(st, tau0_pair, {"I_4x18mT": i_line})})
    return rows


# The zero-bias numbers compared between meshes.
def columns(pt):
    keep = ("barrier_kT", "tau_efold", "tau_dwell", "mean_abs_m", "neighbour_corr", "acf_0p1ns",
            "seeds", "disk_us", "flips")
    return {f: pt[f] for f in keep} | {"peaks_mz": pt["seed_records"][0]["peaks"]}


# The 1 nm runs per anisotropy constant, beside the 2 nm production values (see module
# text).
def mesh_check(stems, step_stems, eq, stale0, steps, rng):
    out = {"reference_2nm": columns(eq["0"]) | {
        "read_change": read_change(stale0), "I_18mT": eq["18"]["I"],
        "tau_18mT_over_cosh": an.cosh_ratio(eq["18"]["tau_efold"], eq["0"]["tau_efold"],
                                            eq["18"]["I"]),
        "steps_72mT": step_rows(steps, eq["0"]["tau_efold"], eq["18"]["I"])}, "by_ku": {}}
    groups, step_groups = {}, {}
    for group, members in ((groups, stems), (step_groups, step_stems)):
        for stem in members:
            group.setdefault(an.meta_of(stem)["ku"], []).append(stem)
    for ku, members in sorted(groups.items()):
        pts, _, series = an.two_state_set(members)
        meta = an.meta_of(members[0])
        zero = pts[0]
        tau0 = zero["tau_efold"][0]
        rec = columns(zero) | {
            "cell_nm": meta["cell"] * 1e9, "pitch_nm": meta["pitch"] * meta["cell"] * 1e9,
            "dt_fs": meta["dt"] * 1e15, "ku": ku,
            "runs": sorted(r["name"] for pt in pts.values() for r in pt["seed_records"])}
        if np.isfinite(tau0):       # a short scan run may end before the read decorrelates
            dt, ss = series[0]
            rec["read_change"] = read_change(an.stale_block(
                ss, dt, zero["p_up"][0], tau0, {"telegraph_efold": zero["tau_efold"]}))
        if 18 in pts:
            rec["I_18mT"] = pts[18]["I"]
            rec["tau_18mT_over_cosh"] = an.cosh_ratio(pts[18]["tau_efold"], zero["tau_efold"],
                                                      pts[18]["I"])
        if ku in step_groups and 18 in pts and np.isfinite(tau0):
            # I at 72 mT is not measured on this mesh: the asymptote of the fixed-asymptote
            # fit is tanh(4 x I(18 mT)), and only the step time is taken from step_set.
            i18 = pts[18]["I"]
            line = {"slope_per_mT": [i18[0] / 18, 0.0], "intercept": [0.0, 0.0],
                    "extrapolation_maps": {"largest_measured": [abs(i18[0]), i18[1]]}}
            recs, _ = an.step_set(step_groups[ku], pts, line, zero["barrier_kT"][0], rng)
            rec["steps_72mT"] = step_rows(recs, zero["tau_efold"], i18)
        out["by_ku"][f"{ku:g}"] = rec
    return out


# Add spin_torque_step.json and mesh_check.json to RECORDS from the check runs.
def main():
    rundirs, records = [Path(a) for a in sys.argv[1:-1]], Path(sys.argv[-1])
    rng = np.random.default_rng(11)
    eq = json.loads((records / "device_p_equilibrium.json").read_text())
    sets = {}
    for stem in an.done_stems(rundirs, ""):
        sets.setdefault(an.meta_of(stem)["set"], []).append(stem)

    if "sttstep" in sets:
        # Equilibrium points: the stored 25 fs spin-torque points and the runs at the step current.
        stored = json.loads((records / "spin_torque.json").read_text())["stt"]["points"]
        points = {float(j): pt for j, pt in stored.items()}
        new, _, _ = an.two_state_set(sets.get("stt", []))
        points |= new
        ref = eq["points"]["0"]
        steps, curves = current_steps(sets["sttstep"], points, ref, rng)
        an.save_json(records / "spin_torque_step.json", {
            "dt_fs": an.PRODUCTION_FS, "tau_c0_ns": ref["tau_efold"],
            "bootstrap_resamples": an.N_BOOT,
            "equilibrium_at_step_current": an.tau_vs_I(new, ref, ref["barrier_kT"][0]),
            "equilibrium_points": an.strip(new), "steps": steps,
            "runs": [dict(an.meta_of(x), wall=Path(f"{x}.done").read_text().strip())
                     for x in sets["sttstep"] + sets.get("stt", [])]})
        np.savez_compressed(records / "spin_torque_step_curves.npz", **curves)

    if "mesh" in sets:
        steps = json.loads((records / "step_response.json").read_text())["steps"]
        rec = mesh_check(sets["mesh"], sets.get("meshstep", []), eq["points"],
                         eq["stale_reads"]["0"], steps, rng)
        rec["runs"] = [dict(an.meta_of(x), wall=Path(f"{x}.done").read_text().strip())
                       for x in sets["mesh"] + sets.get("meshstep", [])]
        an.save_json(records / "mesh_check.json", rec)


if __name__ == "__main__":
    main()
