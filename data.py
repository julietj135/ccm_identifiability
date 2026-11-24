"""Retrieve the observations and metadata"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_data(
    adnimerge_path="/Users/julietjiang/Downloads/ADProjectFiles/AIM1/ADNIMERGE.xlsx",
    pet_path="../UCBERKELEY_PET_CLEAN_ODE.xlsx",
    training_path="../Parameters_PET_TRAIN_ICV.xlsx",
    hyperparams_path="sheets/Identifiability_Hyperparams_loosenbnds.xlsx",
    colormap_path="sheets/Identifiability_Colormap_Figure.xlsx",
    uc_path="sheets/UC_hyperparameters.xlsx",
    subject_params_dir=".",
):
    """Read the workbooks once and return original tables, observations and metrics.

    subjects[rid] holds all PET visits, including visits with missing biomarkers.
    The paired arrays (ab, aab), (pt, apt), and (nn, nnt) hold the available
    observations and their visit ages in workbook order. Metric matrices follow
    df5 row order (metric_rids) and metric_columns order. The optional UC table
    and per-RID parameter workbooks are returned when available.
    """
    df0 = pd.read_excel(adnimerge_path)
    df = pd.read_excel(pet_path)
    df4 = pd.read_excel(training_path)
    df5 = pd.read_excel(hyperparams_path)
    df_fig = pd.read_excel(colormap_path, header=None)
    uc_params = (pd.read_excel(uc_path, index_col="RID")
                 if uc_path is not None and Path(uc_path).is_file() else None)
    df0["datetimes"] = pd.to_datetime(df0.EXAMDATE, format="%m/%d/%Y", errors="coerce")
    df["datetimes"] = pd.to_datetime(df.EXAMDATE, format="%m/%d/%y", errors="coerce")

    raw_ratio = df.HIPPOCAMPUS_VOLUME / df.ICV
    r_lower, r_upper = raw_ratio.quantile([0.05, 0.95])
    AB_lower, AB_upper = df.SUMMARY_SUVR.quantile([0.05, 0.95])
    T_upper = df.META_TEMPORAL_SUVR.quantile(0.95)
    N = 1 - (raw_ratio - r_lower) / (r_upper - r_lower)
    ABETA = (df.SUMMARY_SUVR - AB_lower) / (AB_upper - AB_lower)
    PTAU = df.META_TEMPORAL_SUVR / T_upper

    age = df.groupby("RID").AGE.first().to_dict()
    first_exam = df0.groupby("RID").datetimes.min()
    age_time = first_exam.reindex(df.RID.unique()).to_dict()
    subjects = {}
    for rid, visits in df.groupby("RID", sort=False):
        ages = age[rid] + visits.datetimes.dt.year - age_time[rid].year
        clinical = df0.loc[df0.RID == rid]
        diagnosis = clinical.DX_bl.iloc[0] if len(clinical) else np.nan
        subject = {"PID": visits, "PID_indexes": visits.index, "clinical": clinical,
                   "DX_bl": diagnosis, "visit_ages": ages}
        for series, full_name, obs_name, age_name in (
            (ABETA, "ABETA_PID", "ab", "aab"),
            (PTAU, "PTAU_PID", "pt", "apt"),
            (N, "N_PID", "nn", "nnt"),
        ):
            values = series.loc[visits.index]
            present = values.notna()
            subject[full_name] = values
            subject[obs_name] = values.loc[present].to_numpy()
            subject[age_name] = ages.loc[present].to_numpy()
        subjects[rid] = subject

    metric_columns = ["mean", "median", "std", "range", "max", "min"]
    metric_columns += ["age_" + name for name in metric_columns]
    metric_columns += ["derivative", "count"]
    matrices = [np.empty((len(df5), len(metric_columns))) for _ in range(3)]
    counts = {"CN": [], "MCI": [], "AD": []}
    for i, (_, row) in enumerate(df5.iterrows()):
        rid = row["RID"]
        subject = subjects[rid]
        identifiable = row.iloc[46:55].to_numpy(dtype=float).reshape(3, 3)
        derivatives = row.iloc[61:64].to_numpy(dtype=float)
        baseline = subject["DX_bl"]
        group = baseline if baseline in ("CN", "AD") else "MCI"
        counts[group].append(identifiable.sum())
        subject.update(hyperparams=row.copy(), identifiable=identifiable,
                       derivatives=derivatives, n_id=row["n_ID"],
                       para_A=row.iloc[7:10].to_numpy(dtype=float),
                       para_T=row.iloc[10:13].to_numpy(dtype=float),
                       para_N=row.iloc[13:16].to_numpy(dtype=float),
                       para_AT=row.iloc[7:13].to_numpy(dtype=float))
        for j, (obs_name, age_name) in enumerate(
            [("ab", "aab"), ("pt", "apt"), ("nn", "nnt")]
        ):
            values = subject[obs_name]
            summary = []
            for array in (values, subject[age_name]):
                if array.size:
                    summary.extend((np.mean(array), np.median(array), np.std(array),
                                    np.ptp(array), np.max(array), np.min(array)))
                else:
                    summary.extend([np.nan] * 6)
            matrices[j][i] = [*summary, derivatives[j], len(values)]

    individual_params = {}
    if subject_params_dir is not None:
        for rid in subjects.keys() - set(df5.RID):
            path = Path(subject_params_dir) / f"{rid}_hyperparams.xlsx"
            if path.is_file():
                individual_params[rid] = pd.read_excel(path)

    return {
        "df0": df0, "df": df, "df4": df4, "df5": df5, "df_fig": df_fig,
        "uc_params": uc_params, "individual_params": individual_params,
        "RID": df.RID.unique().tolist(), "RID_train": df4.RID.tolist(),
        "DX": df.DX_closest, "DX_bl": df0.DX_bl, "age": age, "age_time": age_time,
        "hippocampus": df.HIPPOCAMPUS_VOLUME, "ICV": df.ICV, "raw_ratio": raw_ratio,
        "ratio": N, "N": N, "ABETA": ABETA, "PTAU": PTAU,
        "r_lower": r_lower, "r_upper": r_upper, "AB_lower": AB_lower,
        "AB_upper": AB_upper, "T_upper": T_upper, "subjects": subjects,
        "metric_columns": metric_columns, "metric_rids": df5.RID.to_numpy(copy=True),
        "abeta_data": matrices[0], "tau_data": matrices[1], "neuro_data": matrices[2],
        "cn_rid": counts["CN"], "mci_rid": counts["MCI"], "ad_rid": counts["AD"],
    }
