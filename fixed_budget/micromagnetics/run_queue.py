# Author: Manan Gupta <mnn@yogins.com>
"""Run the mumax3 jobs of a run directory through CUDA MPS, several at a time.

Usage: python run_queue.py RUNDIR --gpu 0 --slots 8

Jobs are claimed with an atomic mkdir, so several runners (one per GPU) can
share a directory and a restarted runner skips finished jobs. When a job ends,
its table is packed into RUNDIR/<name>.npz (float16 magnetization) and the
text table is deleted to save disk.
"""

import argparse
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from mmstats import read_table

# Where mumax3, its kernel cache and the MPS pipe live; override with the
# environment variables on another machine.
MX = os.environ.get("MUMAX3_DIR", "/workspace/lmag/mumax/mumax3.12_linux_cuda12.9")
CACHE = os.environ.get("MUMAX3_CACHE", "/workspace/lmag/mumax/cache")
MPS_PIPE = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/workspace/lmag/mps/pipe")


def pack(rundir, name):
    """Convert <name>.out/table.txt to <name>.npz and drop the text table."""
    header, data = read_table(rundir / f"{name}.out" / "table.txt")
    cols = [i for i, h in enumerate(header) if ".region" in h]
    np.savez_compressed(rundir / f"{name}.npz", t=data[:, 0],
                        m=data[:, cols].astype(np.float16),
                        cols=np.array([header[i] for i in cols]))
    (rundir / f"{name}.out" / "table.txt").unlink()


def worker(rundir, names, gpu, log, stop_file=None):
    """Take unclaimed jobs until a full pass finds none (claims freed by others are retried)."""
    env = dict(os.environ, LD_LIBRARY_PATH=f"{MX}/lib",
               CUDA_MPS_PIPE_DIRECTORY=MPS_PIPE)
    claimed = True
    while claimed:
        claimed = False
        for name in names:
            if stop_file and Path(stop_file).exists():
                return
            if (rundir / f"{name}.done").exists():
                continue
            try:
                (rundir / f"{name}.claim").mkdir()
            except FileExistsError:
                continue
            (rundir / f"{name}.claim" / "gpu").write_text(f"{gpu}\n")
            claimed = True
            t0 = time.time()
            with open(rundir / f"{name}.log", "w") as out:
                rc = subprocess.call([f"{MX}/mumax3", "-gpu", str(gpu), "-cache", CACHE, "-f",
                                      str(rundir / f"{name}.mx3")],
                                     stdout=out, stderr=out, env=env)
            if rc == 0:
                pack(rundir, name)
                (rundir / f"{name}.done").write_text(f"{time.time() - t0:.0f} s on gpu {gpu}\n")
            log(f"{name} rc={rc} {time.time() - t0:.0f}s gpu{gpu}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--reclaim", action="store_true", help="free claims of unfinished jobs")
    ap.add_argument("--reverse", action="store_true", help="take the shortest jobs first")
    ap.add_argument("--stop-file", help="stop claiming new jobs once this file exists")
    a = ap.parse_args()
    rundir = Path(a.rundir)
    names = [ln.strip()[:-4] for ln in (rundir / "jobs.txt").read_text().split() if ln.strip()]
    if a.reclaim:
        for n in names:
            if (rundir / f"{n}.claim").exists() and not (rundir / f"{n}.done").exists():
                shutil.rmtree(rundir / f"{n}.claim")
    if a.reverse:
        names = names[::-1]
    lock = threading.Lock()

    def log(msg):
        with lock:
            print(time.strftime("%H:%M:%S"), msg, flush=True)

    threads = [threading.Thread(target=worker, args=(rundir, names, a.gpu, log, a.stop_file))
               for _ in range(a.slots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("queue empty")


if __name__ == "__main__":
    main()
