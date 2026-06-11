#!/usr/bin/env python3
"""Calibrate Re-ID / face cosine thresholds from the LIVE gallery.

Reads the enrolled exemplars (face_exemplars, appearance_exemplars), computes the
distribution of cosine similarity for SAME-identity pairs vs DIFFERENT-identity
pairs, and recommends thresholds that separate them on YOUR data — instead of the
academic defaults in config.py.

    docker exec vms python3 scripts/calibrate_thresholds.py

Vectors are stored 512×float32 little-endian, already L2-normalized, so cosine == dot.
"""
import sqlite3
import sys
import numpy as np

DB = "/app/data/vms.db"


def _load(table):
    con = sqlite3.connect(DB)
    by_id = {}
    for ident, blob in con.execute(f"select identity_id, vector from {table}"):
        v = np.frombuffer(blob, dtype="<f4").astype(np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            by_id.setdefault(ident, []).append(v / n)
    con.close()
    return {k: np.stack(vs) for k, vs in by_id.items() if len(vs) >= 1}


def _pairs(by_id, max_cross=200000):
    same, diff = [], []
    ids = list(by_id)
    for i in ids:                                   # SAME-identity pairs
        M = by_id[i]
        if len(M) >= 2:
            S = M @ M.T
            same.extend(S[np.triu_indices(len(M), k=1)].tolist())
    for a in range(len(ids)):                       # DIFFERENT-identity pairs
        for b in range(a + 1, len(ids)):
            D = (by_id[ids[a]] @ by_id[ids[b]].T).ravel()
            diff.extend(D.tolist())
            if len(diff) > max_cross:
                break
    return np.array(same), np.array(diff)


def _pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def _report(name, by_id, cur_thresh):
    print(f"\n=== {name} ===")
    n_ids = len(by_id)
    n_vec = sum(len(v) for v in by_id.values())
    print(f"личностей с эталонами: {n_ids} | всего эталонов: {n_vec}")
    if n_ids < 2:
        print("  МАЛО ДАННЫХ для кросс-пар (нужно ≥2 личности). Накопи больше идентификаций.")
        return
    same, diff = _pairs(by_id)
    print(f"  пары: свои={len(same)}  чужие={len(diff)}")
    if len(same):
        print(f"  СВОИ  cosine: min={_pct(same,1):.3f} p5={_pct(same,5):.3f} "
              f"медиана={_pct(same,50):.3f} p95={_pct(same,95):.3f}")
    if len(diff):
        print(f"  ЧУЖИЕ cosine: p5={_pct(diff,5):.3f} медиана={_pct(diff,50):.3f} "
              f"p95={_pct(diff,95):.3f} max={_pct(diff,99):.3f}")
    # Recommend: a threshold above almost-all cross pairs but below most same pairs.
    if len(same) and len(diff):
        lo = _pct(diff, 95)         # reject ~95% of impostor pairs above this
        hi = _pct(same, 10)         # keep ~90% of genuine pairs above this
        rec = round((lo + hi) / 2, 3) if hi > lo else round(lo + 0.02, 3)
        sep = "ХОРОШЕЕ разделение" if hi > lo else "ПЕРЕКРЫТИЕ (классы трудно разделить — нужны лучшие эталоны/модель)"
        print(f"  текущий порог: {cur_thresh}")
        print(f"  РЕКОМЕНДАЦИЯ: ~{rec}  ({sep}; импостор-p95={lo:.3f}, генуин-p10={hi:.3f})")


def main():
    try:
        face = _load("face_exemplars")
        app = _load("appearance_exemplars")
    except Exception as e:
        print("ошибка чтения БД:", e); sys.exit(1)
    print("КАЛИБРОВКА ПОРОГОВ ПО ЖИВОЙ ГАЛЕРЕЕ")
    _report("ЛИЦА (face_match_threshold / reid_face_match)", face, "0.45 / 0.42")
    _report("ВНЕШНОСТЬ-ОДЕЖДА (reid_app_match)", app, "0.62 (cross 0.66)")
    print("\nКак применять: выстави рекомендованные значения в .env "
          "(FACE_MATCH_THRESHOLD, REID_FACE_MATCH, REID_APP_MATCH) и перезапусти vms. "
          "Чем больше накоплено личностей/эталонов, тем надёжнее рекомендация.")


if __name__ == "__main__":
    main()
