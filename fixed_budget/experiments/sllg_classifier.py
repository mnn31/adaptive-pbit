# Author: Manan Gupta <mnn@yogins.com>
"""Classifier accuracy with sLLG macrospin neurons against the telegraph model.

Every hidden neuron of a trained 784-256-128-10 network is a macrospin read
as sign(mz). The device is the one characterized in --dev-dir by
experiments/sllg_device.py (device.json): results/sllg_matched/ for the
macrospin matched to the mumax3 disk, results/sllg/ for the 3.5 kT
sensitivity check. r0 is the read interval in units of that device's
tau_c(0), the 1/e time of its zero-bias read autocorrelation. Inference
uses the paper's lane protocol (smtjnn.infer.evaluate_lanes): inputs held
between reads, layers sampled in order each pass, settle mode with a
settle of kappa tau_c(0) from the previous state, streaming with the
run_v2 warm-up rule. All sources run on the full 10,000-image MNIST test
set with the same 500 lanes, so the telegraph column differs from the
paper's records only in the lane count (100 or 200 there).

Sources (kinds):
  sllg        macrospin neurons
  telegraph   the paper's telegraph model (cosh-law rate)
  tcap        telegraph model with the rate growth 2f cosh I capped at a
              constant, the macrospin's relaxation rate after one field
              reversal (TelegraphSource(rate_cap=...)); a diagnostic for
              the streaming gap. Caps come from --dev-dir/device_step.json.
  tlaw_reversal, tlaw_zero
              telegraph model whose rate at each input follows the
              macrospin's measured step-response rate, tau_c(0) over the
              1/e time after a field step to that input, starting from the
              opposite input (reversal) or from zero field (zero); also a
              diagnostic (TelegraphSource(rate_law=...)). Each tabulated
              point sits at the network input that the classifier's
              field-to-I map (device.json) assigns to the step's final field.
  equilibrium independent reads from the macrospin's exact P(+)
  ideal       independent reads from tanh

Networks: "ideal" (ideal-trained seed 0, used for every device sweep),
"na0.2", "na0.5" (telegraph-trained correlation-aware, seed 0; "na0.2s1"
for seed 1 and so on) and "mna0.2", "mna0.5" (trained on this device's
macrospin by experiments/sllg_train.py, in --dev-dir/nets/).

Point groups (--groups): settle, stream, transfer, frontier, cap, legacy
(the point list of the 3.5 kT set). Records go to --dev-dir/clf/, one JSON per run,
skipped if present; telegraph and ideal runs of the shared networks do not
depend on the device and live in results/sllg/clf/. A run whose network
has no checkpoint stops with an error, so the macrospin-trained networks
must be trained before the transfer group (or left out with --nets).
--part table --strict fails if any cell of the matched set is missing.

    python experiments/sllg_classifier.py --dev-dir results/sllg_matched \\
        --part sllg --groups settle stream --gpu cuda:1
    python experiments/sllg_classifier.py --dev-dir results/sllg_matched \\
        --part telegraph --groups settle stream
    python experiments/sllg_classifier.py --dev-dir results/sllg_matched \\
        --part table
"""

import argparse
import csv
import glob
import hashlib
import json
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from smtjnn import sllg
from smtjnn.data import splits
from smtjnn.infer import evaluate_lanes
from smtjnn.sllg import EquilibriumSource, Macrospin, MacrospinSource
from smtjnn.smtj import IdealSource, TelegraphSource
from smtjnn.train import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "results" / "sllg" / "clf"
V2 = ROOT / "results" / "v2"
DEV = ROOT / "results" / "sllg_matched"   # set by --dev-dir

LANES = 500
SETTLE = 4.0
SEEDS = [0, 1, 2]
MAX_SEEDS = 16   # the table reads every noise seed below this

# (network, mode, r0, N, kappa). Settle mode at N = 1 never uses r0 (one
# read right after the settle), so one run stands for every r0.
TRANSFER_NETS = [f"{m}na{r}{s}" for r in ["0.2", "0.5"] for s in
                 ["", "s1", "s2"] for m in ["", "m"]]
GROUPS = {
    "settle": ([("ideal", "settle", 0.02, 1, k) for k in [1, 2, 4, 8]]
               + [("ideal", "settle", 4, 2, SETTLE)]
               + [("ideal", "settle", r0, 32, SETTLE)
                  for r0 in [0.02, 0.2, 4]]),
    "stream": ([("ideal", "stream", r0, 1, SETTLE)
                for r0 in [0.1, 0.2, 0.5, 1, 2, 8]]
               + [("ideal", "stream", r0, 8, SETTLE)
                  for r0 in [0.1, 0.2, 0.5, 1]]
               + [("ideal", "stream", r0, 32, SETTLE)
                  for r0 in [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 8]]
               + [("ideal", "stream", 2, 2, SETTLE),
                  ("ideal", "stream", 1, 4, SETTLE)]),
    "transfer": [(net, "stream", r0, n, SETTLE) for net in TRANSFER_NETS
                 for r0, n in [(0.02, 32), (0.05, 32), (0.1, 32), (0.2, 32),
                               (1, 32), (0.2, 1), (1, 1)]]
                + [(net, "stream", 4, 32, SETTLE) for net in ["na0.2",
                                                             "mna0.2"]],
    "frontier": [(net, "stream", 1, 1, SETTLE)
                 for net in ["na0.5", "na0.5s1", "na0.5s2"]]
                + [("na0.5", "stream", 1, 32, SETTLE)],
    "cap": [("ideal", "stream", r0, 32, SETTLE) for r0 in [0.02, 0.05, 0.1]]
           + [("ideal", "stream", 0.5, 1, SETTLE)],
    # extra grid points near the 90% crossings
    "crossing": [("ideal", "stream", r0, n, SETTLE) for r0, n in
                 [(0.7, 1), (1.5, 1), (0.3, 8), (0.4, 8), (0.07, 32),
                  (0.085, 32), (0.13, 32)]],
    "legacy": ([("ideal", "settle", 0.02, 1, SETTLE)]
               + [("ideal", "settle", r0, 32, SETTLE)
                  for r0 in [0.02, 0.2, 4]]
               + [("ideal", "stream", r0, n, SETTLE) for r0 in [0.1, 0.5, 2]
                  for n in [1, 32]]
               + [("ideal", "stream", r0, 32, SETTLE) for r0 in [0.02, 0.05]]
               + [("ideal", "stream", 8, n, SETTLE) for n in [1, 32]]
               + [("na0.2", "stream", r0, 32, SETTLE) for r0 in [0.02, 1]]),
}
DT_CHECK = [("ideal", "stream", 0.5, 32, SETTLE),
            ("ideal", "stream", 0.1, 1, SETTLE),
            ("ideal", "stream", 0.05, 32, SETTLE),
            ("na0.2", "stream", 0.02, 32, SETTLE)]
# telegraph rate caps, from the step-response rows (from, to) of device_step
CAP_STEPS = {"strong": (-8, 8), "mid": (-4, 4)}
# tabulated telegraph rate laws, from the same rows
LAW_INPUTS = [1, 2, 4, 6, 8]
LAW_STEPS = {"reversal": [(-i, i) for i in LAW_INPUTS],
             "zero": [(0, i) for i in LAW_INPUTS]}
LAW_KINDS = [f"tlaw_{k}" for k in LAW_STEPS]

_DATA = {}


# Path relative to the repo when inside it, absolute otherwise (for the records).
def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


# MNIST splits, loaded once per process
def data():
    if "mnist" not in _DATA:
        _DATA["mnist"] = splits("mnist")
    return _DATA["mnist"]


# Unscored warm-up images per lane in streaming mode, same rule as
# run_v2.lane_params.
def warmup(n, r0):
    return int(min(100, max(10, np.ceil(6.0 / (n * r0)))))


# Checkpoint file of a network name (names are listed in the module docstring).
def net_path(net):
    if net == "ideal":
        return ROOT / "results" / "v2_mnist_seed0.pt"
    m = re.fullmatch(r"(m?)na([\d.]+)(?:s(\d))?", net)
    tr0, seed = m.group(2), int(m.group(3) or 0)
    if m.group(1):
        return DEV / "nets" / f"sllg_na{tr0}_seed{seed}.pt"
    return ROOT / "results" / f"v2_mnist_na{tr0}_seed{seed}.pt"


# the macrospin and its device.json record from the current device directory
def device_record():
    rec = json.loads((DEV / "device.json").read_text())
    d = rec["device"]
    dev = Macrospin(d["delta"], d["moment_per_T"], d["alpha"])
    return dev, rec


# Telegraph rate caps: tau_c(0) over the macrospin's 1/e time after each field
# reversal in CAP_STEPS.
def rate_caps():
    rec = json.loads((DEV / "device_step.json").read_text())
    out = {}
    for key, (a, b) in CAP_STEPS.items():
        row = next(r for r in rec["rows"]
                   if (r["i_from"], r["i_to"]) == (a, b))
        out[key] = {"cap": rec["tau0_s"] / row["tau_1e_s"], "step": [a, b],
                    "i_to_realized": row["i_to_exact"]}
    return out


# Tabulated telegraph rate laws from the macrospin's step responses, one per
# LAW_STEPS set (see the docstring for how the grid is placed).
def rate_laws():
    """Measured rate growth against the nominal input, for each LAW_STEPS set.

    Each point is tau_c(0) over the 1/e time of the ensemble step response
    to that input; the law starts at 1 for I = 0 and is held beyond the
    last point. The steps were run at the fields b_to_T, so the grid "i" is
    b_to_T over the classifier's tesla_per_unit_i: the network input that
    puts the same field on a macrospin neuron.
    """
    rec = json.loads((DEV / "device_step.json").read_text())
    tpi = json.loads((DEV / "device.json").read_text())["tesla_per_unit_i"]
    rows = {(r["i_from"], r["i_to"]): r for r in rec["rows"]}
    out = {}
    for key, steps in LAW_STEPS.items():
        i = [0.0, *(rows[s]["b_to_T"] / tpi for s in steps)]
        out[key] = {"i": i, "i_nominal": [0, *LAW_INPUTS],
                    "growth": [1.0, *(rec["tau0_s"] / rows[s]["tau_1e_s"]
                                      for s in steps)],
                    "cosh_i": np.cosh(i).tolist()}
    return out


# source name used in record file names; capped runs carry their cap
def kind_label(kind, cap):
    return f"tcap{cap:.4g}" if kind == "tcap" else kind


# telegraph and ideal runs of shared networks are device-independent and shared
def record_dir(kind, net):
    if kind in ("telegraph", "ideal") and not net.startswith("m"):
        return SHARED
    return DEV / "clf"


# record file name of one run
def name(kind, net, mode, r0, n, kappa, ns, tag="", cap=None):
    k = "" if kappa == SETTLE else f"_k{kappa:g}"
    return (f"{kind_label(kind, cap)}{tag}_{net}_{mode}_r{r0:g}_T{n}{k}"
            f"_ns{ns}.json")


# code hashes and time, stamped into every record
def provenance():
    files = [Path(sllg.__file__), Path(__file__)]
    return {"code_sha256": {f.name: hashlib.sha256(f.read_bytes())
                            .hexdigest()[:16] for f in files},
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


# One classifier evaluation: build the source for `kind`, run the lane protocol on
# the full test set and write one JSON record. Returns the stored record if it exists.
def run_one(kind, net, mode, r0, n, kappa, ns, gpu="cpu", dt=1e-12, tag="",
            cap=None):
    path = record_dir(kind, net) / name(kind, net, mode, r0, n, kappa, ns,
                                        tag, cap)
    if path.exists():
        return json.loads(path.read_text())
    if not net_path(net).exists():
        raise FileNotFoundError(
            f"no checkpoint for {net} ({net_path(net)}): train it with "
            "experiments/sllg_train.py or leave it out with --nets")
    torch.set_num_threads(4)
    trx, _, _, _, tx, ty = data()
    model = load_checkpoint(net_path(net))
    stream = mode == "stream"
    # only sources with memory need warm-up images in streaming mode
    warm = warmup(n, r0) if stream and kind in ("telegraph", "tcap", "sllg",
                                                *LAW_KINDS) else 0
    law = rate_laws()[kind[5:]] if kind in LAW_KINDS else None
    device_kinds = ("sllg", "equilibrium")
    if kind in device_kinds:
        dev, drec = device_record()
        tpi = drec["tesla_per_unit_i"]
    if kind == "sllg":
        src = MacrospinSource(r0, dev, drec["tau_c0_s"], tpi, dt=dt,
                              streaming=stream, settle_ratio=kappa,
                              noise_seed=ns, device=gpu)
    elif kind in ("telegraph", "tcap", *LAW_KINDS):
        src = TelegraphSource(r0=r0, streaming=stream, settle_ratio=kappa,
                              noise_seed=ns, rate_cap=cap,
                              rate_law=law and (law["i"], law["growth"]))
    elif kind == "equilibrium":
        src = EquilibriumSource(dev, tpi, seed=ns)
    else:
        src = IdealSource(seed=ns)
    t0 = time.time()
    acc = evaluate_lanes(model, tx, ty, src, T=n, lanes=LANES,
                         warmup_x=trx, warmup_per_lane=warm)
    rec = {"source": kind, "net": net, "checkpoint":
           rel(net_path(net)), "mode": mode, "r0": r0,
           "T": n, "settle": kappa, "ns": ns, "acc": acc,
           "n_images": int(tx.shape[0]), "lanes": LANES,
           "warmup_per_lane": warm, "rate_cap": cap, "rate_law": law,
           "dt_s": dt if kind == "sllg" else None,
           "device_dir": rel(DEV) if kind in device_kinds
           or kind in LAW_KINDS or cap or net.startswith("m") else None,
           "tau_c0_s": drec["tau_c0_s"] if kind == "sllg" else None,
           "tesla_per_unit_i": tpi if kind in device_kinds else None,
           "seconds": round(time.time() - t0, 1), **provenance()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    print(path.name, f"{acc:.4f}", f"{rec['seconds']}s", flush=True)
    return rec


# unpack a job tuple for Pool.map
def _job(args):
    return run_one(*args[:7], cap=args[7])


# the (net, mode, r0, N, kappa) points of the chosen groups, without repeats
def points(groups, nets=None):
    pts = []
    for g in groups:
        pts += [p for p in GROUPS[g] if nets is None or p[0] in nets]
    return list(dict.fromkeys(pts))


# All CPU sources (telegraph, capped and tabulated-law telegraph, equilibrium,
# ideal), run in a process pool.
def part_cpu(kind, groups, nets, seeds, workers):
    if kind in ("ideal", "equilibrium"):
        # stateless sources: protocol and r0 do not matter, N does
        jobs = [(kind, "ideal", "iid", 0, n, SETTLE, ns, None)
                for n in [1, 32] for ns in seeds]
        jobs += [(kind, "na0.2", "iid", 0, 32, SETTLE, ns, None)
                 for ns in seeds]
    elif kind == "tcap":
        caps = rate_caps()
        print(json.dumps(caps), flush=True)
        jobs = [("tcap", *p, ns, c["cap"]) for c in caps.values()
                for p in points(groups, nets) for ns in seeds]
    elif kind == "tlaw":
        print(json.dumps(rate_laws()), flush=True)
        jobs = [(k, *p, ns, None) for k in LAW_KINDS
                for p in points(groups, nets) for ns in seeds]
    else:
        jobs = [(kind, *p, ns, None) for p in points(groups, nets)
                for ns in seeds]
    with Pool(workers) as pool:
        pool.map(_job, jobs, chunksize=1)


# device time per image in units of tau_c(0): settle plus reads, or N reads in streaming
def cost(p):
    _, mode, r0, n, kappa = p
    return (kappa + (n - 1) * r0) if mode == "settle" else n * r0


# Noise seeds to run for one macrospin point; the expensive points get only one.
def sllg_seeds(p, seeds):
    """Noise seeds for a macrospin run.

    Extra training seeds (names ending s1, s2) and streaming points needing
    more than 100 tau_c(0) per image get the first noise seed only.
    """
    net, mode = p[0], p[1]
    if re.search(r"s\d$", net) or (mode == "stream" and cost(p) > 100):
        return seeds[:1]
    return seeds


# Macrospin runs on the GPU, cheapest points first so partial runs are useful.
def part_sllg(gpu, groups, nets, seeds, max_cost=np.inf):
    for p in sorted(points(groups, nets), key=cost):
        if cost(p) > max_cost:
            continue
        for ns in sllg_seeds(p, seeds):
            run_one("sllg", *p, ns, gpu=gpu)


# Repeat a few macrospin points at dt = 0.5 ps to check the 1 ps step.
def part_dtcheck(gpu):
    for p in DT_CHECK:
        for ns in SEEDS:
            run_one("sllg", *p, ns, gpu=gpu, dt=0.5e-12, tag="dt0.5ps")


# Pre-activation statistics of the hidden layers and the device P(+) against
# tanh; writes classifier_inputs.json.
def part_inputs():
    dev, drec = device_record()
    tpi = drec["tesla_per_unit_i"]
    i_single_well = dev.b_anis / tpi   # Bz above B_anis: no minority well
    _, _, _, _, tx, _ = data()
    out = {"tesla_per_unit_i": tpi, "i_at_b_anis": i_single_well,
           "networks": {}}
    torch.manual_seed(0)
    for net in ["ideal", "na0.2"]:
        model = load_checkpoint(net_path(net))
        with torch.no_grad():
            i1 = model.layers[0](tx)
            s1 = torch.where(torch.rand_like(i1) < 0.5 * (1 + torch.tanh(i1)),
                             1.0, -1.0)
            i2 = model.layers[1](s1)
        rows = {}
        for layer, i in [(1, i1), (2, i2)]:
            a = i.abs().numpy().ravel()
            rows[layer] = {
                "median_abs_i": float(np.median(a)),
                "p90_abs_i": float(np.quantile(a, 0.9)),
                "p99_abs_i": float(np.quantile(a, 0.99)),
                "max_abs_i": float(a.max()),
                "frac_above_2": float((a > 2).mean()),
                "frac_above_i_at_b_anis": float((a > i_single_well).mean()),
                "max_abs_b_T": float(a.max() * tpi)}
        out["networks"][net] = rows
    grid = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8])
    p_dev = sllg.equilibrium_p_plus(dev, grid * tpi)
    out["activation"] = {"i": grid.tolist(),
                         "p_tanh": (0.5 * (1 + np.tanh(grid))).tolist(),
                         "p_device": p_dev.tolist(),
                         "i_device": np.arctanh(2 * p_dev - 1).tolist()}
    (DEV / "classifier_inputs.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ table
# accuracy list -> mean, SD and count, in percent
def _stats(accs):
    a = 100 * np.array(accs)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1))
            if a.size > 1 else None, "n": int(a.size)}


# Accuracy at this point in the earlier telegraph records (results/v2, 100 or
# 200 lanes), or None.
def _paper(net, mode, r0, n, kappa):
    m = re.fullmatch(r"na([\d.]+)(?:s(\d))?", net)
    if net.startswith("m"):
        return None
    if m:
        pat = (f"e4_mnist_tr{m.group(1)}_s{int(m.group(2) or 0)}_test_"
               f"er{r0:g}_T{n}_ns*.json")
    elif mode == "settle" and kappa != SETTLE:
        pat = f"e2rho_mnist_rho{float(kappa)}_r{r0:g}_T{n}_ns*.json"
    else:
        pat = f"e2_mnist_{mode}_r{r0:g}_T{n}_ns*.json"
    accs = [json.loads(Path(f).read_text())["acc"]
            for f in sorted(glob.glob(str(V2 / pat)))]
    return _stats(accs) if accs else None


# Mean and SD over every noise seed on record for this point.
def _load(kind, p, tag="", cap=None):
    accs = []
    for ns in range(MAX_SEEDS):
        f = record_dir(kind, p[0]) / name(kind, *p, ns, tag, cap)
        if f.exists():
            accs.append(json.loads(f.read_text())["acc"])
    return _stats(accs) if accs else None


# r0 at which accuracy first reaches `level`, with a range from the seed spread.
def crossing(r0s, rows, level=90.0):
    """r0 where accuracy first rises through `level` (linear in log r0).

    Returns the crossing of the seed means and the range spanned by the
    crossings of mean - sd and mean + sd (the seed spread at the two
    bracketing points), with the bracketing grid points.
    """
    r0s = np.asarray(r0s, dtype=float)
    order = np.argsort(r0s)
    r0s = r0s[order]
    mean = np.array([rows[k]["mean"] for k in order])
    sd = np.array([rows[k]["sd"] or 0.0 for k in order])

    # interpolate between the last grid point below level and the first one above
    def cross(acc, log=True):
        above = np.nonzero(acc >= level)[0]
        if above.size == 0 or above[0] == 0:
            return None
        j = above[0]
        x = np.log(r0s) if log else r0s
        f = (level - acc[j - 1]) / (acc[j] - acc[j - 1])
        y = x[j - 1] + f * (x[j] - x[j - 1])
        return float(np.exp(y) if log else y)

    c = cross(mean)
    if c is None:
        return None
    j = int(np.searchsorted(r0s, c))
    return {"r0": c, "r0_linear_interp": cross(mean, log=False),
            "range": sorted(x for x in (cross(mean + sd),
                                        cross(mean - sd)) if x),
            "bracket": [float(r0s[j - 1]), float(r0s[j])],
            "acc_at_bracket": [float(mean[j - 1]), float(mean[j])]}


# Trained-on x tested-on table for the transfer group.
def transfer_summary(rows):
    """Trained-on x tested-on accuracies, mean and SD over training seeds.

    Each training seed enters with the mean over its noise seeds.
    """
    out = []
    for tr0 in ["0.2", "0.5"]:
        for trained in ["telegraph", "macrospin"]:
            base = ("m" if trained == "macrospin" else "") + f"na{tr0}"
            nets = [base, base + "s1", base + "s2"]
            pts = sorted({(r["r0"], r["T"]) for r in rows
                          if r["group"] == "transfer" and r["net"] == base})
            for r0, n in pts:
                cell = {"train_r0": float(tr0), "trained_on": trained,
                        "r0": r0, "T": n}
                for col, tested in [("telegraph", "telegraph"),
                                    ("sllg", "macrospin"),
                                    *((k, k) for k in LAW_KINDS)]:
                    per_net = [r[col]["mean"] for r in rows
                               if r["group"] == "transfer"
                               and r["net"] in nets and r["r0"] == r0
                               and r["T"] == n and r.get(col)]
                    cell[f"tested_on_{tested}"] = (_stats_pct(per_net)
                                                   if per_net else None)
                out.append(cell)
    return out


# How much of the telegraph-to-macrospin accuracy gap each diagnostic closes.
def gap_closed(rows, cols, min_gap=2.0):
    """Share of the telegraph-to-macrospin gap that each diagnostic closes.

    (telegraph - diagnostic) / (telegraph - macrospin) at every streaming
    point of the ideal-trained network where the gap is at least min_gap
    points; 0 is the cosh-law telegraph model, 1 the macrospin.
    """
    out, seen = [], set()
    for r in rows:
        key = (r["r0"], r["T"])
        if (r["net"] != "ideal" or r["mode"] != "stream" or key in seen
                or not (r.get("sllg") and r.get("telegraph"))):
            continue
        gap = r["telegraph"]["mean"] - r["sllg"]["mean"]
        # small gaps make the share too noisy to read
        if gap < min_gap:
            continue
        shares = {c: (r["telegraph"]["mean"] - r[c]["mean"]) / gap
                  for c in cols if r.get(c)}
        if shares:
            seen.add(key)
            out.append({"r0": r["r0"], "T": r["T"], "gap_points": gap,
                        "macrospin": r["sllg"]["mean"],
                        "telegraph": r["telegraph"]["mean"],
                        **{c: r[c]["mean"] for c in shares},
                        "share_closed": shares})
    return sorted(out, key=lambda g: (g["T"], g["r0"]))


# values already in percent -> mean, SD and number of training seeds
def _stats_pct(values):
    a = np.array(values)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1))
            if a.size > 1 else None, "n_training_seeds": int(a.size)}


# Cells the matched record set should hold but does not.
def missing_cells(rows, caps, laws):
    need = {"sllg": set(GROUPS) - {"legacy"},
            "telegraph": set(GROUPS) - {"legacy"},
            **{f"tcap_{k}": {"cap", "stream", "crossing"} for k in caps},
            **{f"tlaw_{k}": set(GROUPS) - {"legacy", "settle"}
               for k in laws}}
    have = {(r["group"], r["net"], r["mode"], r["r0"], r["T"], r["settle"])
            : r for r in rows}
    out = []
    for group, pts in GROUPS.items():
        for p in pts:
            row = have.get((group, *p), {})
            out += [f"{col} {group} {p}" for col, groups in need.items()
                    if group in groups and not row.get(col)]
    for r in rows:
        if r["group"] == "iid":
            out += [f"{c} iid {r['net']} N={r['T']}"
                    for c in ("equilibrium", "ideal") if not r.get(c)]
    done = {(r["net"], r["mode"], r["r0"], r["T"], r["settle"])
            for r in rows if r["group"] == "dt0.5ps"}
    out += [f"sllg dt0.5ps {p}" for p in DT_CHECK if p not in done]
    return out


# Collect every record into classifier_table.json and .csv, with the 90%
# crossings, the gap shares and the transfer table; --strict fails on missing cells.
def part_table(strict=False):
    have_step = (DEV / "device_step.json").exists()
    caps = rate_caps() if have_step else {}
    laws = rate_laws() if have_step else {}
    rows = []
    for group, pts in GROUPS.items():
        for p in pts:
            net, mode, r0, n, kappa = p
            row = {"group": group, "net": net, "mode": mode, "r0": r0,
                   "T": n, "settle": kappa, "sllg": _load("sllg", p),
                   "telegraph": _load("telegraph", p),
                   "paper": _paper(*p)}
            for key, c in caps.items():
                row[f"tcap_{key}"] = _load("tcap", p, cap=c["cap"])
            for key in laws:
                row[f"tlaw_{key}"] = _load(f"tlaw_{key}", p)
            if any(row[k] for k in ("sllg", "telegraph")):
                rows.append(row)
    for net, n in [("ideal", 1), ("ideal", 32), ("na0.2", 32)]:
        p = (net, "iid", 0, n, SETTLE)
        rows.append({"group": "iid", "net": net, "mode": "iid", "r0": None,
                     "T": n, "settle": None,
                     "equilibrium": _load("equilibrium", p),
                     "ideal": _load("ideal", p)})
    for p in DT_CHECK:
        s = _load("sllg", p, "dt0.5ps")
        if s:
            rows.append({"group": "dt0.5ps", "net": p[0], "mode": p[1],
                         "r0": p[2], "T": p[3], "settle": p[4], "sllg": s})
    stream = {}
    for r in rows:
        if (r["group"] in ("stream", "crossing") and r["net"] == "ideal"
                and r["mode"] == "stream"):
            stream.setdefault(r["T"], []).append(r)
    cross = {}
    for n, rs in sorted(stream.items()):
        for col in ["sllg", "telegraph", *LAW_KINDS]:
            ok = [r for r in rs if r.get(col)]
            if len(ok) >= 2:
                cross[f"{col}_N{n}"] = crossing([r["r0"] for r in ok],
                                                [r[col] for r in ok])
    diag = [f"tcap_{k}" for k in caps] + [f"tlaw_{k}" for k in laws]
    out = {"device_dir": rel(DEV), "rate_caps": caps,
           "rate_laws": laws, "gap_closed": gap_closed(rows, diag),
           "crossings_90": cross, "transfer_2x2": transfer_summary(rows),
           "missing": missing_cells(rows, caps, laws), "rows": rows}
    (DEV / "classifier_table.json").write_text(json.dumps(out, indent=1))
    cols = ["sllg", "telegraph", *diag, "paper", "equilibrium", "ideal"]
    with open(DEV / "classifier_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "net", "mode", "r0", "N", "settle"] +
                   [f"{c}_{s}" for c in cols for s in ("mean", "sd", "n")])
        for r in rows:
            vals = []
            for c in cols:
                s = r.get(c)
                vals += ([round(s["mean"], 3), s["sd"] and round(s["sd"], 3),
                          s["n"]] if s else ["", "", ""])
            w.writerow([r["group"], r["net"], r["mode"], r["r0"], r["T"],
                        r["settle"]] + vals)
    for r in rows:
        txt = "  ".join(f"{c}={r[c]['mean']:.2f}+-{r[c]['sd'] or 0:.2f}"
                        for c in cols if r.get(c))
        print(f"{r['group']:8s} {r['net']:8s} {r['mode']:6s} r0={r['r0']} "
              f"N={r['T']} k={r['settle']}: {txt}")
    print(json.dumps({"rate_caps": caps, "rate_laws": laws,
                      "crossings_90": cross}, indent=1))
    for g in out["gap_closed"]:
        txt = "  ".join(f"{c}={v:.2f}" for c, v in g["share_closed"].items())
        print(f"gap closed, N={g['T']} r0={g['r0']} "
              f"({g['gap_points']:.1f} points): {txt}")
    for c in out["transfer_2x2"]:
        txt = "  ".join(
            f"{k[10:]}={c[k]['mean']:.2f}+-{c[k]['sd'] or 0:.2f}"
            f"(n={c[k]['n_training_seeds']})"
            for k in c if k.startswith("tested_on_") and c[k])
        print(f"r0_train={c['train_r0']} trained on {c['trained_on']:9s} "
              f"r0={c['r0']} N={c['T']}: {txt}")
    if strict and out["missing"]:
        raise SystemExit("missing cells:\n  " + "\n  ".join(out["missing"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["ideal", "equilibrium", "telegraph", "tcap",
                             "tlaw", "sllg", "dtcheck", "inputs", "table"])
    ap.add_argument("--dev-dir", default=str(DEV))
    ap.add_argument("--groups", nargs="*", default=["settle", "stream"],
                    choices=list(GROUPS))
    ap.add_argument("--nets", nargs="*", default=None,
                    help="restrict to these networks")
    ap.add_argument("--gpu", default="cuda:1")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--strict", action="store_true",
                    help="table: fail if a cell of the matched set is "
                         "missing")
    ap.add_argument("--max-cost", type=float, default=np.inf,
                    help="sllg: skip points needing more than this many "
                         "tau_c(0) of device time per image")
    args = ap.parse_args()
    DEV = Path(args.dev_dir).resolve()
    t0 = time.time()
    if args.part in ("ideal", "equilibrium", "telegraph", "tcap", "tlaw"):
        part_cpu(args.part, args.groups, args.nets, args.seeds, args.workers)
    elif args.part == "sllg":
        part_sllg(args.gpu, args.groups, args.nets, args.seeds,
                  args.max_cost)
    elif args.part == "dtcheck":
        part_dtcheck(args.gpu)
    elif args.part == "inputs":
        part_inputs()
    else:
        part_table(args.strict)
    print(f"{args.part} done in {time.time() - t0:.0f}s", flush=True)
