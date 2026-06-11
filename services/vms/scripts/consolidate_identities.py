#!/usr/bin/env python3
"""Consolidate over-split person identities: merge duplicates of the same person.

The Re-ID gallery can over-split one person into many identities (faceless / back /
low-light crops give moderate appearance cosine that straddles the match threshold,
and a polluted gallery makes every new sighting "ambiguous" → a new identity). This
collapses them: cluster person-identities by appearance-centroid cosine (union-find)
and merge each cluster into its lowest id via the live merge endpoint.

    docker exec vms python3 scripts/consolidate_identities.py [--threshold 0.62] [--dry-run]
"""
import argparse, json, sqlite3, sys, urllib.request
import numpy as np

DB = "/app/data/vms.db"
BASE = "http://127.0.0.1:8120"


def centroids():
    c = sqlite3.connect(DB)
    ids = [r[0] for r in c.execute("select id from identities where object_class='person'")]
    by = {}
    for iid, blob in c.execute(
        "select a.identity_id, a.vector from appearance_exemplars a "
        "join identities i on i.id=a.identity_id where i.object_class='person'"):
        v = np.frombuffer(blob, dtype="<f4").astype("float32")
        by.setdefault(iid, []).append(v / (np.linalg.norm(v) + 1e-9))
    cent = {}
    for iid, vs in by.items():
        m = np.mean(np.stack(vs), axis=0)
        cent[iid] = m / (np.linalg.norm(m) + 1e-9)
    c.close()
    return ids, cent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.62)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ids, cent = centroids()
    items = sorted(cent.items())
    keys = [k for k, _ in items]
    M = np.stack([v for _, v in items]) if items else np.zeros((0, 512))
    print(f"person-личностей: {len(ids)} | с appearance-центроидом: {len(keys)} | порог склейки: {a.threshold}")
    if len(keys) < 2:
        print("нечего схлопывать"); return

    parent = {k: k for k in keys}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)   # keep the lowest id as survivor

    S = M @ M.T
    n = len(keys)
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= a.threshold:
                union(keys[i], keys[j])

    clusters = {}
    for k in keys:
        clusters.setdefault(find(k), []).append(k)
    to_merge = {t: [s for s in srcs if s != t] for t, srcs in clusters.items() if len(srcs) > 1}
    n_dupes = sum(len(v) for v in to_merge.values())
    print(f"кластеров для слияния: {len(to_merge)} | поглощается дублей: {n_dupes} "
          f"-> станет ~{len(clusters)} уникальных")
    if a.dry_run:
        for t, srcs in list(to_merge.items())[:10]:
            print(f"  #{t} <- {srcs[:8]}{'…' if len(srcs)>8 else ''}")
        return

    ok = 0
    for target, sources in to_merge.items():
        try:
            body = json.dumps({"target_id": int(target), "source_ids": [int(s) for s in sources]}).encode()
            req = urllib.request.Request(BASE + "/api/identities/merge", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=60).read()
            ok += 1
        except Exception as e:
            print(f"  ! merge #{target} failed: {e}")
    c = sqlite3.connect(DB)
    left = c.execute("select count(*) from identities where object_class='person'").fetchone()[0]
    c.close()
    print(f"\nслияний выполнено: {ok}/{len(to_merge)} | person-личностей осталось: {left}")


if __name__ == "__main__":
    main()
