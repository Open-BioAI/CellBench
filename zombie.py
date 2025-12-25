import argparse
import time
import torch

def boost(device: int, util: float, interval: float, mat: int, fp16: bool):
    torch.cuda.set_device(device)
    dtype = torch.float16 if fp16 else torch.float32

    a = torch.randn(mat, mat, device=f"cuda:{device}", dtype=dtype)
    b = torch.randn(mat, mat, device=f"cuda:{device}", dtype=dtype)

    # warmup
    for _ in range(10):
        _ = a @ b
    torch.cuda.synchronize()

    burn_t = interval * util
    sleep_t = interval * (1 - util)

    while True:
        t0 = time.time()
        # burn phase
        while (time.time() - t0) < burn_t:
            _ = a @ b
        torch.cuda.synchronize()
        if sleep_t > 0:
            time.sleep(sleep_t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default="0")
    ap.add_argument("--util", type=float, default=0.2, help="extra duty-cycle, e.g. 0.2 ~= +20% busy time")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--mat", type=int, default=4096)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = []
    for d in gpus:
        p = ctx.Process(target=boost, args=(d, args.util, args.interval, args.mat, args.fp16), daemon=True)
        p.start()
        procs.append(p)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=2)

if __name__ == "__main__":
    main()
