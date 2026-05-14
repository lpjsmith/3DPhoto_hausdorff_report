# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# """
# Repeatability + within-mesh ICC for NEW landmarks.

# - Scans *_landmark*.json (case-insensitive)
# - subject_id (mesh) = normalized filename stem (remove trailing _rgF/_landmarks)
# - run index = integer in the top-level folder name (e.g., SNH1 -> 1)
# - Uses NEW landmarks: new_nasion, new_lh_coord, new_rh_coord

# Outputs in OUT_DIR:
#   - landmarks_raw.csv
#   - landmarks_long.csv
#   - subjects_repeat_counts.csv
#   - subjects_repeat_matrix.csv
#   - jitter_per_subject.csv                (per mesh & landmark)
#   - jitter_rc_summary.csv                 (pooled RC per landmark + combined)
#   - pairwise_repeatability_per_subject.csv
#   - pairwise_rc_summary.csv
#   - icc_within_mesh.csv                   (per mesh)
#   - icc_within_mesh_summary.csv           (summary across meshes)
# """

# import json, re
# from pathlib import Path
# from typing import List, Dict, Tuple, Optional
# import numpy as np
# import pandas as pd

# # --------- CONFIG ---------
# BASE_DIR = r"/mnt/c/Users/klay.luke.PSYDUCK/Desktop/properly trimmed-2dl/SNH/SNH NEW"
# OUT_DIR  = BASE_DIR + "/out"
# EXCLUDE_NASION_FROM_ICC = True  # nasion is [0,0,0] in new_*; exclude from ICC by default

# # --------- DISCOVERY ---------
# def find_landmark_jsons(base: Path) -> List[Path]:
#     pat = re.compile(r'(?i)[_-]landmarks?.*\.json$')
#     return [p for p in base.rglob("*.json") if p.is_file() and pat.search(p.name)]

# # --------- SUBJECT/RUN ---------
# def subject_id_from_stem(stem: str) -> str:
#     """
#     Normalize the filename stem so repeats across SNH1/2/3 map to the same mesh.
#     Removes trailing '_landmarks' / '-landmarks' and '_rgF' / '-rgF'.
#     Leaves the rest intact (e.g., 'Charite1_AC_SNHPUC_20250129_1').
#     """
#     s = re.sub(r'(?i)[_-]landmarks?$', '', stem)
#     s = re.sub(r'(?i)[_-]rgf?$', '', s)
#     return s

# def run_from_top_folder(rel_parts: Tuple[str, ...]) -> Optional[int]:
#     if not rel_parts:
#         return None
#     m = re.search(r'(\d+)', rel_parts[0])
#     return int(m.group(1)) if m else None

# # --------- JSON HELPERS ---------
# def _deep_find_key(obj, key):
#     if isinstance(obj, dict):
#         if key in obj:
#             return obj[key]
#         for v in obj.values():
#             hit = _deep_find_key(v, key)
#             if hit is not None:
#                 return hit
#     elif isinstance(obj, list):
#         for it in obj:
#             hit = _deep_find_key(it, key)
#             if hit is not None:
#                 return hit
#     return None

# def _first_present(data, *keys):
#     for k in keys:
#         v = _deep_find_key(data, k)
#         if v is not None:
#             return v
#     return None

# # --------- EXTRACTION ---------
# def extract_rows(landmarks_fp: Path, base: Path) -> List[Dict]:
#     with open(landmarks_fp, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     try:
#         rel = landmarks_fp.relative_to(base)
#         rel_parts = rel.parts
#     except Exception:
#         rel_parts = ()

#     run = run_from_top_folder(rel_parts)
#     stem = landmarks_fp.stem
#     subject_id = subject_id_from_stem(stem)

#     dt_str = _first_present(data, "datetime", "Datetime", "dateTime")
#     mtime = landmarks_fp.stat().st_mtime

#     # keys = {
#     #     "new_nasion": "nasion",
#     #     "new_lh_coord": "left",
#     #     "new_rh_coord": "right",
#     # }

#     keys = {
#         "initial_nasion": "nasion",
#         "initial_lh_coord": "left",
#         "initial_rh_coord": "right",
#     }


#     rows: List[Dict] = []
#     for jkey, lname in keys.items():
#         coords = _deep_find_key(data, jkey)
#         if not (isinstance(coords, (list, tuple)) and len(coords) == 3):
#             continue
#         try:
#             x, y, z = map(float, coords)
#         except Exception:
#             continue
#         rows.append({
#             "top_folder": rel_parts[0] if len(rel_parts) > 0 else "",
#             "landmarks_path": str(landmarks_fp),
#             "file_stem": stem,
#             "subject_id": subject_id,     # mesh id
#             "run": run,                   # 1..n from SNH*
#             "dt_str": str(dt_str) if dt_str is not None else None,
#             "mtime": mtime,
#             "landmark": lname,            # nasion/left/right
#             "x": x, "y": y, "z": z,
#         })
#     return rows

# # --------- DIAGNOSTICS ---------
# def write_repeat_counts(df: pd.DataFrame, outdir: Path) -> None:
#     counts = (df[["subject_id","landmarks_path"]]
#               .drop_duplicates()
#               .groupby("subject_id")
#               .size()
#               .reset_index(name="n_files"))
#     counts = counts.sort_values(["n_files","subject_id"], ascending=[False, True])
#     counts_csv = outdir / "subjects_repeat_counts.csv"
#     counts.to_csv(counts_csv, index=False)
#     print(f"[OK] Wrote: {counts_csv}")
#     print("[INFO] Top subjects by #files:")
#     print(counts.head(20).to_string(index=False))

# def write_repeat_matrix(df: pd.DataFrame, outdir: Path) -> None:
#     mat = (df[["subject_id","run"]]
#            .drop_duplicates()
#            .assign(has=1)
#            .pivot_table(index="subject_id", columns="run", values="has", fill_value=0))
#     csv = outdir / "subjects_repeat_matrix.csv"
#     mat.to_csv(csv)
#     print(f"[OK] Wrote: {csv}")

# # --------- STATS: ICC ---------
# def icc_3_1(matrix: np.ndarray) -> float:
#     """
#     ICC(3,1): two-way mixed, consistency, single measures.
#     matrix shape: (n_items, n_raters) with no missing values.
#     """
#     n, k = matrix.shape
#     if n < 2 or k < 2:
#         return np.nan
#     mean_item = matrix.mean(axis=1, keepdims=True)
#     mean_rater = matrix.mean(axis=0, keepdims=True)
#     grand_mean = matrix.mean()
#     ss_item = k * np.sum((mean_item - grand_mean) ** 2)
#     ss_rater = n * np.sum((mean_rater - grand_mean) ** 2)
#     ss_total = np.sum((matrix - grand_mean) ** 2)
#     ss_error = ss_total - ss_item - ss_rater
#     ms_item = ss_item / (n - 1)
#     ms_error = ss_error / ((n - 1) * (k - 1))
#     return (ms_item - ms_error) / (ms_item + (k - 1) * ms_error)

# def per_mesh_icc(df_long: pd.DataFrame, exclude_nasion: bool=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     rows = []
#     for sid, sub in df_long.groupby("subject_id"):
#         S = sub.copy()
#         if exclude_nasion:
#             S = S[S["landmark"] != "nasion"]
#         S["item"] = S["landmark"] + "." + S["axis"]  # e.g., left.x
#         piv = S.pivot_table(index="item", columns="run", values="value", aggfunc="mean")
#         piv = piv.dropna(axis=0, how="any")
#         n_items, k = piv.shape
#         icc_val = icc_3_1(piv.to_numpy()) if (n_items >= 2 and k >= 2) else np.nan
#         rows.append({"subject_id": sid, "n_items": int(n_items), "k_repeats": int(k), "icc_3_1_within_mesh": icc_val})
#     out = pd.DataFrame(rows)

#     def fisher_z_mean(series: pd.Series) -> float:
#         x = series.dropna()
#         if x.empty: return np.nan
#         x = x.clip(-0.999999, 0.999999)
#         z = np.arctanh(x)
#         return float(np.tanh(np.mean(z)))

#     summary = pd.DataFrame({
#         "mean_ICC": [out["icc_3_1_within_mesh"].mean(skipna=True)],
#         "median_ICC": [out["icc_3_1_within_mesh"].median(skipna=True)],
#         "fisher_z_mean_ICC": [fisher_z_mean(out["icc_3_1_within_mesh"])],
#         "n_meshes": [out["icc_3_1_within_mesh"].notna().sum()]
#     })
#     return out, summary

# # --------- STATS: frame-invariant repeatability ---------
# def pooled_within_sd(sd_list: List[float], n_runs_list: List[int]) -> float:
#     """
#     Pooled within-subject SD across meshes:
#       sqrt( sum_i ( (n_i-1) * SD_i^2 ) / sum_i (n_i-1) )
#     Only uses meshes with SD_i finite and n_i >= 2.
#     """
#     num, den = 0.0, 0
#     for sd, n in zip(sd_list, n_runs_list):
#         if n is None or n < 2 or sd is None or not np.isfinite(sd):
#             continue
#         num += (n - 1) * (sd ** 2)
#         den += (n - 1)
#     return np.sqrt(num / den) if den > 0 else np.nan

# def rc_from_pooled_sd(pooled_sd: float) -> float:
#     return 1.96 * np.sqrt(2.0) * pooled_sd if np.isfinite(pooled_sd) else np.nan

# # --------- PIPELINE ---------
# def run_pipeline(BASE_DIR: str, OUT_DIR: str) -> None:
#     base = Path(BASE_DIR)
#     outdir = Path(OUT_DIR)
#     outdir.mkdir(parents=True, exist_ok=True)

#     files = find_landmark_jsons(base)
#     print(f"[INFO] Base: {base}")
#     print(f"[INFO] Found {len(files)} landmark JSON(s).")
#     for p in files[:5]:
#         print(f"[INFO]  e.g. {p}")

#     all_rows: List[Dict] = []
#     for lfp in files:
#         try:
#             all_rows.extend(extract_rows(lfp, base))
#         except Exception as e:
#             print(f"[WARN] Could not read {lfp}: {e}")

#     print(f"[INFO] Extracted {len(all_rows)} landmark row(s) from {len(files)} file(s).")
#     if not all_rows:
#         print("[ERROR] No landmark rows extracted.")
#         return

#     df = pd.DataFrame(all_rows)

#     # Save raw + long
#     raw_csv = outdir / "landmarks_raw.csv"
#     df.to_csv(raw_csv, index=False)
#     df_long = df.melt(
#         id_vars=["top_folder","landmarks_path","file_stem","subject_id","run","landmark"],
#         value_vars=["x","y","z"], var_name="axis", value_name="value"
#     )
#     long_csv = outdir / "landmarks_long.csv"
#     df_long.to_csv(long_csv, index=False)

#     # Diagnostics
#     write_repeat_counts(df, outdir)
#     write_repeat_matrix(df, outdir)

#     # ---------- 3D JITTER per mesh & landmark ----------
#     means = (df.groupby(["subject_id","landmark"], as_index=False)[["x","y","z"]]
#                .mean().rename(columns={"x":"mx","y":"my","z":"mz"}))
#     dfm = df.merge(means, on=["subject_id","landmark"], how="left")
#     dfm["jitter"] = np.sqrt((dfm["x"]-dfm["mx"])**2 + (dfm["y"]-dfm["my"])**2 + (dfm["z"]-dfm["mz"])**2)

#     jitter_subj = (dfm.groupby(["subject_id","landmark"])
#                       .agg(n_runs=("jitter","count"),
#                            jitter_mean=("jitter","mean"),
#                            jitter_sd=("jitter", lambda s: float(np.std(s, ddof=1)) if len(s)>=2 else np.nan),
#                            jitter_rms=("jitter", lambda s: float(np.sqrt(np.mean(np.square(s)))))
#                            )
#                       .reset_index())
#     jitter_csv = outdir / "jitter_per_subject.csv"
#     jitter_subj.to_csv(jitter_csv, index=False)

#     # Pooled RC per landmark + combined
#     rows = []
#     for lm in sorted(df["landmark"].unique()):
#         sub = jitter_subj[jitter_subj["landmark"] == lm]
#         pooled_sd = pooled_within_sd(sub["jitter_sd"].tolist(), sub["n_runs"].tolist())
#         rows.append({"scope":"per_landmark", "landmark":lm,
#                      "pooled_within_sd":pooled_sd, "RC":rc_from_pooled_sd(pooled_sd)})
#     pooled_sd_all = pooled_within_sd(jitter_subj["jitter_sd"].tolist(), jitter_subj["n_runs"].tolist())
#     rows.append({"scope":"combined_all", "landmark":"all",
#                  "pooled_within_sd":pooled_sd_all, "RC":rc_from_pooled_sd(pooled_sd_all)})
#     jitter_rc = pd.DataFrame(rows)
#     jitter_rc_csv = outdir / "jitter_rc_summary.csv"
#     jitter_rc.to_csv(jitter_rc_csv, index=False)

#     # ---------- Inter-landmark distance repeatability ----------
#     pivot = df.pivot_table(index=["subject_id","run"], columns="landmark", values=["x","y","z"])
#     pair_rows = []
#     for (sid, run), block in pivot.groupby(level=[0,1]):
#         try:
#             L = np.array([block[("x","left")], block[("y","left")], block[("z","left")]]).squeeze()
#             R = np.array([block[("x","right")], block[("y","right")], block[("z","right")]]).squeeze()
#             N = np.array([block[("x","nasion")], block[("y","nasion")], block[("z","nasion")]]).squeeze()
#         except KeyError:
#             continue
#         pair_rows.extend([
#             {"subject_id": sid, "run": int(run), "pair": "L-R", "dist": float(np.linalg.norm(L - R))},
#             {"subject_id": sid, "run": int(run), "pair": "L-N", "dist": float(np.linalg.norm(L - N))},
#             {"subject_id": sid, "run": int(run), "pair": "R-N", "dist": float(np.linalg.norm(R - N))},
#         ])
#     pair_df = pd.DataFrame(pair_rows)
#     pair_stats = (pair_df.groupby(["subject_id","pair"])
#                     .agg(n_runs=("dist","count"),
#                          mean=("dist","mean"),
#                          sd=("dist", lambda s: float(np.std(s, ddof=1)) if len(s)>=2 else np.nan),
#                          rms=("dist", lambda s: float(np.sqrt(np.mean(np.square(s)))))
#                          ).reset_index())
#     pair_subj_csv = outdir / "pairwise_repeatability_per_subject.csv"
#     pair_stats.to_csv(pair_subj_csv, index=False)

#     prow = []
#     for p in ["L-R","L-N","R-N"]:
#         sub = pair_stats[pair_stats["pair"] == p]
#         pooled_sd = pooled_within_sd(sub["sd"].tolist(), sub["n_runs"].tolist())
#         prow.append({"scope":"per_pair", "pair":p,
#                      "pooled_within_sd":pooled_sd, "RC":rc_from_pooled_sd(pooled_sd)})
#     pooled_sd_all_pairs = pooled_within_sd(pair_stats["sd"].tolist(), pair_stats["n_runs"].tolist())
#     prow.append({"scope":"combined_all_pairs", "pair":"all",
#                  "pooled_within_sd":pooled_sd_all_pairs, "RC":rc_from_pooled_sd(pooled_sd_all_pairs)})
#     pair_rc = pd.DataFrame(prow)
#     pair_rc_csv = outdir / "pairwise_rc_summary.csv"
#     pair_rc.to_csv(pair_rc_csv, index=False)

#     # ---------- Within-mesh ICC (items = axes; raters = runs) ----------
#     icc_df, icc_summary = per_mesh_icc(df_long, exclude_nasion=EXCLUDE_NASION_FROM_ICC)
#     icc_mesh_csv = outdir / "icc_within_mesh.csv"
#     icc_df.to_csv(icc_mesh_csv, index=False)
#     icc_mesh_summary_csv = outdir / "icc_within_mesh_summary.csv"
#     icc_summary.to_csv(icc_mesh_summary_csv, index=False)

#     # ---------- Done ----------
#     print(f"[OK] Wrote: {raw_csv}")
#     print(f"[OK] Wrote: {long_csv}")
#     print(f"[OK] Wrote: {jitter_csv}")
#     print(f"[OK] Wrote: {jitter_rc_csv}")
#     print(f"[OK] Wrote: {pair_subj_csv}")
#     print(f"[OK] Wrote: {pair_rc_csv}")
#     print(f"[OK] Wrote: {icc_mesh_csv}")
#     print(f"[OK] Wrote: {icc_mesh_summary_csv}")

# if __name__ == "__main__":
#     run_pipeline(BASE_DIR, OUT_DIR)


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Only ICC + jitter tables, for BOTH initial_* and new_* landmarks.

- Scans *_landmark*.json (case-insensitive)
- subject_id = normalized filename stem (removes trailing _rgF/_landmarks)
- run        = integer in top-level folder name (e.g., SNH1 -> 1)

Outputs in OUT_DIR:
  - jitter_per_subject_initial.csv
  - jitter_per_subject_new.csv
  - icc_within_mesh_initial.csv
  - icc_within_mesh_summary_initial.csv
  - icc_within_mesh_new.csv
  - icc_within_mesh_summary_new.csv
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

# ======== CONFIG ========
BASE_DIR = r"/mnt/c/Users/klay.luke.PSYDUCK/Desktop/properly trimmed-2dl/MHT/MHT NEW"
OUT_DIR  = BASE_DIR + "/out"
EXCLUDE_NASION_FROM_ICC = True  # recommended (new_nasion is fixed at [0,0,0])

# ======== DISCOVERY ========
def find_landmark_jsons(base: Path) -> List[Path]:
    pat = re.compile(r'(?i)[_-]landmarks?.*\.json$')
    return [p for p in base.rglob("*.json") if p.is_file() and pat.search(p.name)]

# ======== SUBJECT/RUN PARSING ========
def subject_id_from_stem(stem: str) -> str:
    """
    Normalize the filename stem so repeats across SNH1/2/3 map to the same mesh.
    Removes trailing '_landmarks' / '-landmarks' and '_rgF' / '-rgF'.
    Leaves the rest intact (e.g., 'Charite1_AC_SNHPUC_20250129_1').
    """
    s = re.sub(r'(?i)[_-]landmarks?$', '', stem)
    s = re.sub(r'(?i)[_-]rgf?$', '', s)
    return s

def run_from_top_folder(rel_parts: Tuple[str, ...]) -> Optional[int]:
    if not rel_parts:
        return None
    m = re.search(r'(\d+)', rel_parts[0])
    return int(m.group(1)) if m else None

# ======== JSON HELPERS ========
def _deep_find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = _deep_find_key(v, key)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for it in obj:
            hit = _deep_find_key(it, key)
            if hit is not None:
                return hit
    return None

# ======== EXTRACTION (BOTH initial & new) ========
def extract_rows_both(landmarks_fp: Path, base: Path) -> List[Dict]:
    with open(landmarks_fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        rel = landmarks_fp.relative_to(base)
        rel_parts = rel.parts
    except Exception:
        rel_parts = ()

    run = run_from_top_folder(rel_parts)
    stem = landmarks_fp.stem
    subject_id = subject_id_from_stem(stem)

    rows: List[Dict] = []
    for which, keys in [
        ("initial", {"initial_nasion": "nasion", "initial_lh_coord": "left", "initial_rh_coord": "right"}),
        ("new",     {"new_nasion":     "nasion", "new_lh_coord":     "left", "new_rh_coord":     "right"}),
    ]:
        for jkey, lname in keys.items():
            coords = _deep_find_key(data, jkey)
            if not (isinstance(coords, (list, tuple)) and len(coords) == 3):
                continue
            try:
                x, y, z = map(float, coords)
            except Exception:
                continue
            rows.append({
                "which": which,                 # 'initial' or 'new'
                "landmarks_path": str(landmarks_fp),
                "file_stem": stem,
                "subject_id": subject_id,       # mesh id
                "run": run,                     # 1..n from SNH*
                "landmark": lname,              # nasion/left/right
                "x": x, "y": y, "z": z,
            })
    return rows

# ======== ICC(3,1) ========
def icc_3_1(matrix: np.ndarray) -> float:
    """
    ICC(3,1): two-way mixed, consistency, single measures.
    matrix shape: (n_items, n_raters) with no missing values.
    """
    n, k = matrix.shape
    if n < 2 or k < 2:
        return np.nan
    mean_item = matrix.mean(axis=1, keepdims=True)
    mean_rater = matrix.mean(axis=0, keepdims=True)
    grand_mean = matrix.mean()
    ss_item = k * np.sum((mean_item - grand_mean) ** 2)
    ss_rater = n * np.sum((mean_rater - grand_mean) ** 2)
    ss_total = np.sum((matrix - grand_mean) ** 2)
    ss_error = ss_total - ss_item - ss_rater
    ms_item = ss_item / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    return (ms_item - ms_error) / (ms_item + (k - 1) * ms_error)

def per_mesh_icc(df_long: pd.DataFrame, exclude_nasion: bool=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for sid, sub in df_long.groupby("subject_id"):
        S = sub.copy()
        if exclude_nasion:
            S = S[S["landmark"] != "nasion"]
        # 'item' = landmark.axis (e.g., left.x)
        S["item"] = S["landmark"] + "." + S["axis"]
        piv = S.pivot_table(index="item", columns="run", values="value", aggfunc="mean")
        piv = piv.dropna(axis=0, how="any")
        n_items, k = piv.shape
        icc_val = icc_3_1(piv.to_numpy()) if (n_items >= 2 and k >= 2) else np.nan
        rows.append({"subject_id": sid, "n_items": int(n_items), "k_repeats": int(k), "icc_3_1_within_mesh": icc_val})
    out = pd.DataFrame(rows)

    def fisher_z_mean(series: pd.Series) -> float:
        x = series.dropna()
        if x.empty: return np.nan
        x = x.clip(-0.999999, 0.999999)
        z = np.arctanh(x)
        return float(np.tanh(np.mean(z)))

    summary = pd.DataFrame({
        "mean_ICC": [out["icc_3_1_within_mesh"].mean(skipna=True)],
        "median_ICC": [out["icc_3_1_within_mesh"].median(skipna=True)],
        "fisher_z_mean_ICC": [fisher_z_mean(out["icc_3_1_within_mesh"])],
        "n_meshes": [out["icc_3_1_within_mesh"].notna().sum()]
    })
    return out, summary

# ======== JITTER (frame-invariant) ========
def compute_jitter_tables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-mesh/per-landmark jitter table:
      subject_id, landmark, n_runs, jitter_mean, jitter_sd, jitter_rms
    """
    means = (df.groupby(["subject_id","landmark"], as_index=False)[["x","y","z"]]
               .mean().rename(columns={"x":"mx","y":"my","z":"mz"}))
    dfm = df.merge(means, on=["subject_id","landmark"], how="left")
    dfm["jitter"] = np.sqrt((dfm["x"]-dfm["mx"])**2 + (dfm["y"]-dfm["my"])**2 + (dfm["z"]-dfm["mz"])**2)

    jitter_subj = (dfm.groupby(["subject_id","landmark"])
                      .agg(n_runs=("jitter","count"),
                           jitter_mean=("jitter","mean"),
                           jitter_sd=("jitter", lambda s: float(np.std(s, ddof=1)) if len(s)>=2 else np.nan),
                           jitter_rms=("jitter", lambda s: float(np.sqrt(np.mean(np.square(s)))))
                           ).reset_index())
    return jitter_subj

# ======== PIPELINE ========
def run(BASE_DIR: str, OUT_DIR: str) -> None:
    base = Path(BASE_DIR)
    outdir = Path(OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    files = find_landmark_jsons(base)
    print(f"[INFO] Base: {base}")
    print(f"[INFO] Found {len(files)} landmark JSON(s).")

    all_rows: List[Dict] = []
    for lfp in files:
        try:
            all_rows.extend(extract_rows_both(lfp, base))
        except Exception as e:
            print(f"[WARN] Could not read {lfp}: {e}")

    if not all_rows:
        print("[ERROR] No landmark rows extracted.")
        return

    df_all = pd.DataFrame(all_rows)

    # ----- Split initial vs new -----
    for which in ["initial", "new"]:
        d = df_all[df_all["which"] == which].copy()
        if d.empty:
            print(f"[WARN] No rows for '{which}' landmarks; skipping.")
            continue

        # JITTER
        jitter_tbl = compute_jitter_tables(d)
        jitter_csv = Path(OUT_DIR) / f"jitter_per_subject_{which}.csv"
        jitter_tbl.to_csv(jitter_csv, index=False)

        # ICC within mesh: build long form (without writing to disk)
        d_long = d.melt(
            id_vars=["landmarks_path","file_stem","subject_id","run","landmark"],
            value_vars=["x","y","z"], var_name="axis", value_name="value"
        )
        icc_df, icc_summary = per_mesh_icc(d_long, exclude_nasion=EXCLUDE_NASION_FROM_ICC)

        icc_csv = Path(OUT_DIR) / f"icc_within_mesh_{which}.csv"
        icc_df.to_csv(icc_csv, index=False)
        icc_sum_csv = Path(OUT_DIR) / f"icc_within_mesh_summary_{which}.csv"
        icc_summary.to_csv(icc_sum_csv, index=False)

        print(f"[OK] Wrote: {jitter_csv}")
        print(f"[OK] Wrote: {icc_csv}")
        print(f"[OK] Wrote: {icc_sum_csv}")

if __name__ == "__main__":
    run(BASE_DIR, OUT_DIR)
