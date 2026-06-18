"""聚合 72-cell sweep 结果：按 (embodiment, magnitude, flag) 分组算 ζ 均值与漂移效果。"""
import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_list(s):
    if not s:
        return []
    return json.loads(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    args = ap.parse_args()
    sweep_dir = Path(args.sweep_dir)
    csv_path = sweep_dir / "sweep_matrix.csv"
    rows = list(csv.DictReader(csv_path.open()))
    print(f"[INFO] {len(rows)} rows from {csv_path}")

    # 每行的 zeta_runtime_mean 是 4 axis list，取均值
    for r in rows:
        zlist = parse_list(r["zeta_runtime_over_nominal_mean"])
        r["_zeta_mean_scalar"] = statistics.mean(zlist) if zlist else float("nan")
        r["_zeta_max_scalar"] = max(parse_list(r["zeta_runtime_over_nominal_max"]) or [0])

    # 1) per (embodiment, magnitude, flag)
    agg_emag = {}
    for r in rows:
        key = (r["_embodiment"], r["_magnitude"], r["_flag"])
        agg_emag.setdefault(key, []).append(r)

    print()
    print("=== Per (embodiment, magnitude, flag) ===")
    print(f"{'embodiment':<16}{'mag':>6}{'flag':>16}{'n':>4}{'zeta_mean':>12}{'pe_active':>12}{'angmax':>10}")
    for (emb, mag, flag), bucket in sorted(agg_emag.items()):
        zmean = statistics.mean([r["_zeta_mean_scalar"] for r in bucket])
        pe = statistics.mean([float(r["pe_active_ratio"]) for r in bucket])
        angmax = statistics.mean([float(r["ang_vel_norm_max"]) for r in bucket])
        print(f"{emb:<16}{mag:>6}{flag:>16}{len(bucket):>4}{zmean:>12.4f}{pe:>12.4f}{angmax:>10.3f}")

    # 2) tune_gains vs identity_init 总览（按 magnitude 分组）
    print()
    print("=== Per (magnitude, flag) ===")
    print(f"{'mag':>6}{'flag':>16}{'n':>4}{'zeta_mean':>12}{'pe_active':>12}{'angmax':>10}")
    agg_mf = {}
    for r in rows:
        key = (r["_magnitude"], r["_flag"])
        agg_mf.setdefault(key, []).append(r)
    for (mag, flag), bucket in sorted(agg_mf.items()):
        zmean = statistics.mean([r["_zeta_mean_scalar"] for r in bucket])
        pe = statistics.mean([float(r["pe_active_ratio"]) for r in bucket])
        angmax = statistics.mean([float(r["ang_vel_norm_max"]) for r in bucket])
        print(f"{mag:>6}{flag:>16}{len(bucket):>4}{zmean:>12.4f}{pe:>12.4f}{angmax:>10.3f}")

    # 3) tune vs identity 总比较
    print()
    print("=== Tune_gains vs Identity_init (all 36 cells each) ===")
    for flag in ("tune_gains", "identity_init"):
        bucket = [r for r in rows if r["_flag"] == flag]
        zmean = statistics.mean([r["_zeta_mean_scalar"] for r in bucket])
        pe = statistics.mean([float(r["pe_active_ratio"]) for r in bucket])
        angmax = statistics.mean([float(r["ang_vel_norm_max"]) for r in bucket])
        wallt = statistics.mean([float(r["_wall_seconds"]) for r in bucket])
        print(f"  {flag:<14} n={len(bucket):2d}  zeta_mean={zmean:.4f}  pe_active={pe:.4f}  angmax={angmax:.3f}  wall={wallt:.1f}s")

    # 4) embodiment 总览
    print()
    print("=== Per embodiment (all 18 cells each) ===")
    for emb in ("base", "long_body", "heavy_moderate", "asymmetric"):
        bucket = [r for r in rows if r["_embodiment"] == emb and r["_flag"] == "tune_gains"]
        if not bucket:
            continue
        zmean = statistics.mean([r["_zeta_mean_scalar"] for r in bucket])
        zmax = max([r["_zeta_max_scalar"] for r in bucket])
        pe = statistics.mean([float(r["pe_active_ratio"]) for r in bucket])
        print(f"  [tune_gains] {emb:<16} n={len(bucket):2d}  zeta_mean={zmean:.4f}  zeta_max={zmax:.4f}  pe={pe:.4f}")

    # 写入 summary.json
    out = {
        "n_rows": len(rows),
        "tune_vs_identity": {
            flag: {
                "n": len([r for r in rows if r["_flag"] == flag]),
                "zeta_mean": statistics.mean([r["_zeta_mean_scalar"] for r in rows if r["_flag"] == flag]),
                "pe_active_ratio": statistics.mean([float(r["pe_active_ratio"]) for r in rows if r["_flag"] == flag]),
                "ang_vel_max_mean": statistics.mean([float(r["ang_vel_norm_max"]) for r in rows if r["_flag"] == flag]),
            }
            for flag in ("tune_gains", "identity_init")
        },
        "per_embodiment_tune_gains": {
            emb: {
                "zeta_mean": statistics.mean([r["_zeta_mean_scalar"] for r in rows if r["_embodiment"] == emb and r["_flag"] == "tune_gains"]),
                "zeta_max": max([r["_zeta_max_scalar"] for r in rows if r["_embodiment"] == emb and r["_flag"] == "tune_gains"]),
                "pe_active_ratio": statistics.mean([float(r["pe_active_ratio"]) for r in rows if r["_embodiment"] == emb and r["_flag"] == "tune_gains"]),
            }
            for emb in ("base", "long_body", "heavy_moderate", "asymmetric")
        },
    }
    out_path = sweep_dir / "summary_aggregated.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[INFO] wrote {out_path}")


if __name__ == "__main__":
    main()
