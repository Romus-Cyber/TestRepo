# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 06:49:57 2025

@Author : RPMRS & CGPT
"""

# -*- coding: utf-8 -*-
"""
12f_gridsearch_bigmove_hyperXL_v3h_safe2.py

Safeguards vs v3h_safe:
- Skip combos when *labels* for (B,H) are missing (logged once per (B,H)).
- Stronger fuzzy matcher: accepts both `B50` and `50p` style, and `hit_up_50p_H1440`.
- Bullet‑proof `safe_log()` (never crashes the main loop on Windows).
- Clear device forcing: `--device cpu` disables CuPy entirely.
- CPU fallback if a CuPy kernel fails mid‑run (recasts arrays to NumPy).
- tqdm preserved and accurate even when skipping many combos.

Usage example (PowerShell):
python 12f_gridsearch_bigmove_hyperXL_v3h_safe2.py `
  --pair AUDUSD `
  --preds_path "C:\\...\\AUDUSD_preds_bigmove.parquet" `
  --labels_path "C:\\...\\AUDUSD_labels_bigmove.parquet" `
  --out_dir "C:\\...\\gridXL_bigmove_GPU0\\AUDUSD" `
  --horizons "1440,2880,10080" --barriers "50,100,200,500" `
  --prob_thres_long  "0.55:0.95:0.05" `
  --prob_thres_short "0.55:0.95:0.05" `
  --entry_policies "every_H,first_of_block,window_max" `
  --conflict_policies "argmax,long_only,short_only" `
  --sizing "fixed,ev" --risk_mult "0.25:2.50:0.25" --size_cap "1:6:1" `
  --costs_pips "1.5:3.0:0.5" --sl_frac "0.1:1.0:0.1" `
  --dtype float16 --batch_combos 524288 --max_combos 150000000 `
  --flush_every 524288 --resume --device cuda:0
"""
import os, csv, math, re, argparse, sys
from pathlib import Path

import pandas as pd
import numpy as np

CUPY_OK = False
try:
    import cupy as cp  # noqa: F401
    CUPY_OK = True
except Exception:
    cp = None
    CUPY_OK = False

from tqdm import tqdm

# ---------------------- utils ----------------------
def safe_log(path: Path, msg: str):
    """Never crash on logging. Best‑effort append."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Normalize newlines & strip non‑printables that Windows hates
        clean = (msg or "").replace("\r\n", "\n").replace("\r", "\n")
        clean = clean.encode("utf-8", errors="replace").decode("utf-8", errors="ignore")
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(clean.rstrip("\n") + "\n")
    except Exception as e:
        # As absolute last resort, print to stderr
        try:
            sys.stderr.write(f"[LOG-FAIL] {e}\n")
        except Exception:
            pass


def _is_range_spec(txt: str) -> bool:
    return ":" in str(txt)


def parse_num_list(txt: str, as_int=False):
    """Accepts scalar, CSV or inclusive range a:b:c."""
    if isinstance(txt, (int, float)):
        return [int(txt) if as_int else float(txt)]
    s = str(txt).strip()
    if _is_range_spec(s):
        a, b, c = s.split(":")
        a = float(a); b = float(b); c = float(c)
        if c == 0:
            vals = [a]
        else:
            n = int(math.floor((b - a) / c + 1e-12)) + 1
            vals = [a + i * c for i in range(max(n, 0))]
            vals = [float(f"{v:.10f}") for v in vals if v <= b + 1e-12]
        return [int(v) for v in vals] if as_int else vals
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return [int(p) if as_int else float(p) for p in parts]
    # scalar
    return [int(s) if as_int else float(s)]


def parse_str_list(txt: str):
    return [t.strip() for t in str(txt).split(",") if t.strip()]


# ---------------- fuzzy matching columns ----------------
_non_alnum = re.compile(r"[^a-z0-9]+")

def _norm(s: str) -> str:
    return _non_alnum.sub("", str(s).lower())


def _h_tag(H):
    return f"h{int(H)}"


def _b_tags(B):
    """Return alternative tags for the same barrier value: ['b50','50p']"""
    if float(B).is_integer():
        bi = int(B)
        return [f"b{bi}", f"{bi}p"]
    btxt = str(B).replace(".", "")
    return [f"b{btxt}", f"{btxt}p"]


def _dir_tokens(up=True, is_pred=False):
    base = ["prob", "proba", "pred"] if is_pred else ["hit", "touch", "reached", "label", "target"]
    return base + (["up", "long", "buy"] if up else ["down", "short", "sell"])  # direction words


def find_best_col(all_cols, B, H, up=True, is_pred=False):
    want = set(_dir_tokens(up, is_pred))
    ht = _h_tag(H)
    best = None
    best_score = -1
    sample = []

    for c in all_cols:
        cn = _norm(c)
        sample.append(c)
        score = 0
        # direction tokens
        for t in want:
            if t in cn:
                score += 1
        # horizon tag
        if ht in cn or f"h{int(H)}" in cn:
            score += 3
        # barrier tags (accept bXX or XXp forms)
        for bt in _b_tags(B):
            if bt in cn:
                score += 3
        # boost if explicit naming pattern seen often
        if ("hit" in cn or "reached" in cn or "touch" in cn) and ("up" in cn or "down" in cn):
            score += 1
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        return None, sample[:10]

    cn = _norm(best)
    okH = (ht in cn) or (f"h{int(H)}" in cn)
    okB = any(bt in cn for bt in _b_tags(B))

    if not (okH and okB):
        return None, sample[:10]

    if is_pred and not any(t in cn for t in ["prob", "proba", "pred"]):
        return None, sample[:10]
    if (not is_pred) and not any(t in cn for t in ["hit", "touch", "reached", "label", "target"]):
        return None, sample[:10]

    return best, sample[:10]


# ---------------- device selection ----------------

def pick_xp(device: str, dtype: str):
    d = (dtype or "float32").lower()
    if d not in ("float16", "float32", "float64"):
        d = "float32"

    # Force CPU if requested
    if str(device).lower().startswith("cpu"):
        xp = np
        on_gpu = False
        dt = {"float16": np.float16, "float32": np.float32, "float64": np.float64}[d]
        print("[DEVICE] Using CPU / NumPy")
        return xp, dt, on_gpu

    # Else try GPU
    if CUPY_OK:
        try:
            dev_id = 0
            if ":" in str(device):
                dev_id = int(str(device).split(":")[1])
            import cupy as cp  # local import to avoid globals when CPU forced
            cp.cuda.Device(dev_id).use()
            mempool = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(mempool.malloc)
            xp = cp
            on_gpu = True
            dt = {"float16": cp.float16, "float32": cp.float32, "float64": cp.float64}[d]
            print(f"[DEVICE] Using GPU {dev_id} / CuPy {d}")
            return xp, dt, on_gpu
        except Exception as e:
            print(f"[DEVICE] GPU init failed -> CPU fallback ({e})")

    # CPU fallback
    xp = np
    on_gpu = False
    dt = {"float16": np.float16, "float32": np.float32, "float64": np.float64}[d]
    print("[DEVICE] Using CPU / NumPy (fallback)")
    return xp, dt, on_gpu


# ---------------- core logic ----------------

def build_combos(Hs, Bs, thrL, thrS, entries, conflicts, sizings, rmults, caps, costs, sl_fracs):
    for H in Hs:
        for B in Bs:
            for tl in thrL:
                for ts in thrS:
                    for e in entries:
                        for c in conflicts:
                            for s in sizings:
                                for r in rmults:
                                    for cap in caps:
                                        for ct in costs:
                                            for slf in sl_fracs:
                                                yield (H, B, tl, ts, e, c, s, r, cap, ct, slf)


def apply_conflict_policy(xp, p_up, p_down, thrL, thrS, policy):
    long_mask  = p_up   >= thrL
    short_mask = p_down >= thrS

    if policy == "long_only":
        direction = xp.where(long_mask, 1, 0)
    elif policy == "short_only":
        direction = xp.where(short_mask, -1, 0)
    else:
        both = long_mask & short_mask
        choose_long  = both & (p_up >= p_down)
        choose_short = both & (p_down >  p_up)
        direction = xp.where(choose_long, 1,
                     xp.where(choose_short, -1,
                     xp.where(long_mask, 1,
                     xp.where(short_mask, -1, 0))))
    return direction.astype(getattr(xp, "int8", np.int8))


def entry_filter_indices(total_len, H, policy):
    if policy == "first_of_block":
        return slice(0, total_len, int(H))
    else:
        return slice(0, total_len, 1)


def _to_numpy(arr):
    try:
        import cupy as cp  # noqa: F401
        if hasattr(arr, "get"):
            return arr.get()
    except Exception:
        pass
    return np.asarray(arr)


def eval_combo(xp, dt, p_up, p_down, hit_up, hit_down, H, B, thrL, thrS,
               entry_pol, confl_pol, sizing, risk_mult, size_cap,
               costs_pips, sl_frac, on_gpu, err_log: Path):
    """Evaluate a single combo. If a CuPy kernel fails, fall back to CPU for this combo."""
    try:
        return _eval_combo_impl(xp, dt, p_up, p_down, hit_up, hit_down, H, B, thrL, thrS,
                                entry_pol, confl_pol, sizing, risk_mult, size_cap,
                                costs_pips, sl_frac)
    except Exception as e:
        # GPU hiccup? Recast to NumPy and retry once
        safe_log(err_log, f"[GPU‑fallback] {e}")
        try:
            nu  = _to_numpy(p_up)
            nd  = _to_numpy(p_down)
            nhu = _to_numpy(hit_up).astype(bool)
            nhd = _to_numpy(hit_down).astype(bool)
            return _eval_combo_impl(np, np.float32, nu, nd, nhu, nhd, H, B, thrL, thrS,
                                    entry_pol, confl_pol, sizing, risk_mult, size_cap,
                                    costs_pips, sl_frac)
        except Exception as e2:
            safe_log(err_log, f"[EVAL-FAIL] {e2}")
            return {
                "trades": 0, "wins": 0, "win_rate": 0.0,
                "sum_pips_raw": 0.0, "avg_pips_raw": 0.0, "sum_pips_net": 0.0
            }


def _eval_combo_impl(xp, dt, p_up, p_down, hit_up, hit_down, H, B, thrL, thrS,
                     entry_pol, confl_pol, sizing, risk_mult, size_cap,
                     costs_pips, sl_frac):
    n = p_up.shape[0]
    idx = entry_filter_indices(n, H, entry_pol)

    pu_s = p_up[idx]
    pd_s = p_down[idx]
    hu_s = hit_up[idx]
    hd_s = hit_down[idx]

    direction = apply_conflict_policy(xp, pu_s, pd_s, thrL, thrS, confl_pol)
    has_trade = (direction != 0)

    if hasattr(xp, "count_nonzero"):
        trades = int(xp.count_nonzero(has_trade))
    else:
        trades = int(np.count_nonzero(has_trade))

    if trades == 0:
        return {
            "trades": 0, "wins": 0, "win_rate": 0.0,
            "sum_pips_raw": 0.0, "avg_pips_raw": 0.0, "sum_pips_net": 0.0
        }

    zero = xp.zeros_like(pu_s, dtype=dt)
    payoff = xp.array(zero, dtype=dt)

    long_idx  = (direction == 1)
    short_idx = (direction == -1)

    payoff = xp.where(long_idx,  xp.where(hu_s,  B, -sl_frac * B), payoff)
    payoff = xp.where(short_idx, xp.where(hd_s,  B, -sl_frac * B), payoff)

    wins_mask = (long_idx & hu_s) | (short_idx & hd_s)
    if hasattr(xp, "count_nonzero"):
        wins = int(xp.count_nonzero(wins_mask))
    else:
        wins = int(np.count_nonzero(wins_mask))
    win_rate = wins / trades if trades > 0 else 0.0

    # sizing
    if sizing == "fixed":
        size = xp.where(has_trade, risk_mult, 0.0).astype(dt)
    else:
        conf = xp.abs(pu_s - pd_s)
        m = float(conf.max()) if getattr(conf, "size", 0) else 1.0
        conf = conf / (m + 1e-9)
        # clip
        if hasattr(xp, "clip"):
            conf = xp.clip(conf, 0, 1)
        else:
            conf = np.clip(conf, 0, 1)
        size = (conf * risk_mult).astype(dt)
        size = xp.where(has_trade, size, 0.0)

    # cap
    if hasattr(xp, "minimum"):
        size = xp.minimum(size, size_cap)
    else:
        size = np.minimum(size, size_cap)

    # sums
    ones = getattr(xp, "ones_like", np.ones_like)(has_trade, dtype=dt)
    mask = xp.where(has_trade, ones, 0.0)

    raw_sum = float((payoff * mask).sum().item() if hasattr(payoff, "sum") else np.sum(payoff * mask))
    avg_raw = raw_sum / trades if trades > 0 else 0.0

    net_sum  = float(((payoff * size) - (costs_pips * size)).sum().item() if hasattr(payoff, "sum") else np.sum((payoff * size) - (costs_pips * size)))

    return {
        "trades": trades,
        "wins": wins,
        "win_rate": win_rate,
        "sum_pips_raw": raw_sum,
        "avg_pips_raw": avg_raw,
        "sum_pips_net": net_sum,
    }


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--preds_path", required=True)
    ap.add_argument("--labels_path", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--horizons", required=True)
    ap.add_argument("--barriers", required=True)
    ap.add_argument("--prob_thres_long", required=True)
    ap.add_argument("--prob_thres_short", required=True)
    ap.add_argument("--entry_policies", default="every_H")
    ap.add_argument("--conflict_policies", default="argmax")
    ap.add_argument("--sizing", default="fixed")
    ap.add_argument("--risk_mult", default="1.0")
    ap.add_argument("--size_cap", default="3.0")
    ap.add_argument("--costs_pips", default="1.0")
    ap.add_argument("--sl_frac", default="0.5")

    ap.add_argument("--dtype", choices=["float16","float32","float64"], default="float16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_combos", type=int, default=65536)
    ap.add_argument("--max_combos", type=int, default=0)
    ap.add_argument("--rows_subsample", type=int, default=0)
    ap.add_argument("--flush_every", type=int, default=50000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sort_by", default="sum_pips_net")
    ap.add_argument("--skip_missing_labels", action="store_true", default=True)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "grid_results.csv"
    err_log = out_dir / "errors.log"

    xp, dt, on_gpu = pick_xp(args.device, args.dtype)

    print(f"\n[LOAD] preds:  {args.preds_path}")
    df_preds = pd.read_parquet(args.preds_path)
    print(f"[LOAD] labels: {args.labels_path}")
    df_labels = pd.read_parquet(args.labels_path)

    # align on __dt__ if present
    if "__dt__" in df_preds.columns and "__dt__" in df_labels.columns:
        df_preds = df_preds.sort_values("__dt__")
        df_labels = df_labels.sort_values("__dt__")
        df = pd.merge(df_preds, df_labels, on="__dt__", how="inner", suffixes=("","_lab"))
        print(f"[INFO] Rows aligned: {df.shape[0]:,}")
    else:
        n = min(len(df_preds), len(df_labels))
        df = pd.concat([df_preds.iloc[:n].reset_index(drop=True),
                        df_labels.iloc[:n].reset_index(drop=True)], axis=1)
        print(f"[INFO] Rows aligned by length: {n:,}")

    if args.rows_subsample and args.rows_subsample > 0:
        df = df.iloc[::args.rows_subsample].reset_index(drop=True)
        print(f"[INFO] Rows after subsample(step={args.rows_subsample}): {len(df):,}")

    all_cols = list(map(str, df.columns))

    # parse grids
    Hs       = parse_num_list(args.horizons, as_int=True)
    Bs       = parse_num_list(args.barriers,  as_int=False)
    thrL     = parse_num_list(args.prob_thres_long,  as_int=False)
    thrS     = parse_num_list(args.prob_thres_short, as_int=False)
    entries  = parse_str_list(args.entry_policies)
    conflicts= parse_str_list(args.conflict_policies)
    sizings  = parse_str_list(args.sizing)
    rmults   = parse_num_list(args.risk_mult, as_int=False)
    caps     = parse_num_list(args.size_cap,  as_int=False)
    costs    = parse_num_list(args.costs_pips,as_int=False)
    sl_fracs = parse_num_list(args.sl_frac,   as_int=False)

    total = (len(Hs)*len(Bs)*len(thrL)*len(thrS)*
             len(entries)*len(conflicts)*len(sizings)*
             len(rmults)*len(caps)*len(costs)*len(sl_fracs))
    total_use = min(total, args.max_combos) if args.max_combos and args.max_combos > 0 else total
    print(f"[GRID] Combos totaux={total:,}" + (f" → limité à {total_use:,}" if total_use < total else ""))

    # CSV init
    write_header = True
    if args.resume and out_csv.exists() and out_csv.stat().st_size > 0:
        write_header = False

    f = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    header = [
        "pair","H","B","prob_thres_long","prob_thres_short",
        "entry_policy","conflict_policy","sizing","risk_mult","size_cap",
        "costs_pips","sl_frac",
        "trades","wins","win_rate",
        "sum_pips_raw","avg_pips_raw","sum_pips_net"
    ]
    if write_header:
        w.writerow(header); f.flush()

    processed = 0
    batch_rows = []
    pbar = tqdm(total=total_use, desc=f"Grid[{args.pair}]")

    cache_cols = {}     # (B,H) -> (p_up, p_down, hit_up, hit_down) or None if missing
    missing_once = set()  # remember which (B,H) were missing labels to avoid spam

    def load_bh_arrays(B, H):
        key = (float(B), int(H))
        if key in cache_cols:
            return cache_cols[key]

        # find prediction columns first (mandatory)
        pu_col, smp1 = find_best_col(all_cols, B, H, up=True,  is_pred=True)
        pd_col, smp2 = find_best_col(all_cols, B, H, up=False, is_pred=True)
        if pu_col is None or pd_col is None:
            cache_cols[key] = None
            if key not in missing_once:
                safe_log(err_log, f"[preds] colonnes manquantes pour B{B} H{H}. samples={smp1+smp2}")
                missing_once.add(key)
            return None

        # labels may be missing for some (B,H) → skip gracefully
        hu_col, smp3 = find_best_col(all_cols, B, H, up=True,  is_pred=False)
        hd_col, smp4 = find_best_col(all_cols, B, H, up=False, is_pred=False)
        if hu_col is None or hd_col is None:
            cache_cols[key] = None
            if key not in missing_once and args.skip_missing_labels:
                safe_log(err_log, f"[labels] colonnes manquantes pour B{B} H{H}. samples={smp3+smp4}")
                missing_once.add(key)
            return None

        # cast on CPU to float32 for stability then to target / device
        pu_cpu = df[str(pu_col)].to_numpy(dtype=np.float32)
        pd_cpu = df[str(pd_col)].to_numpy(dtype=np.float32)
        hu_cpu = df[str(hu_col)].astype(bool).to_numpy()
        hd_cpu = df[str(hd_col)].astype(bool).to_numpy()

        if xp is np:
            p_up = pu_cpu.astype(dt)
            p_dn = pd_cpu.astype(dt)
            h_up = hu_cpu
            h_dn = hd_cpu
        else:
            import cupy as cp  # local import
            p_up = cp.asarray(pu_cpu, dtype=dt)
            p_dn = cp.asarray(pd_cpu, dtype=dt)
            h_up = cp.asarray(hu_cpu)
            h_dn = cp.asarray(hd_cpu)

        cache_cols[key] = (p_up, p_dn, h_up, h_dn)
        return cache_cols[key]

    gen = build_combos(Hs, Bs, thrL, thrS, entries, conflicts, sizings, rmults, caps, costs, sl_fracs)

    for combo in gen:
        if processed >= total_use:
            break
        (H,B,tl,ts,e,c,s,r,cap,ct,slf) = combo

        arrs = load_bh_arrays(B, H)
        if arrs is None:
            # Skip whole bundle of thresholds for this (B,H) silently
            processed += 1
            pbar.update(1)
            continue

        (p_up, p_down, hit_up, hit_down) = arrs
        # eval with safety
        res = eval_combo(xp, dt, p_up, p_down, hit_up, hit_down,
                         H, B, float(tl), float(ts),
                         e, c, s, float(r), float(cap), float(ct), float(slf),
                         on_gpu, err_log)

        row = [
            args.pair, int(H), float(B), float(tl), float(ts),
            e, c, s, float(r), float(cap),
            float(ct), float(slf),
            int(res["trades"]), int(res["wins"]), float(res["win_rate"]),
            float(res["sum_pips_raw"]), float(res["avg_pips_raw"]), float(res["sum_pips_net"])
        ]
        batch_rows.append(row)
        processed += 1
        if (processed % args.flush_every) == 0:
            w.writerows(batch_rows); f.flush(); batch_rows.clear()
        pbar.update(1)

    if batch_rows:
        w.writerows(batch_rows); f.flush(); batch_rows.clear()

    f.close()
    pbar.close()
    print(f"[DONE] Résultats → {out_csv}")


if __name__ == "__main__":
    main()
