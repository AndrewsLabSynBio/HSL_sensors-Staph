#!/usr/bin/env python3
"""
Sensor attribute statistics: one-way ANOVA + Tukey HSD

Expected input columns:
    Host
    HSL
    QSR
    Promoter
    Replicate   (optional)
    Kd
    n
    ymin
    ymax
    Fold activation
    LOD
    Sensitivity

Analysis:
- PFtsH, PBla, and PVeg are compared within each Host/HSL/QSR/Attribute group.
- Kd, ymin, ymax, Fold activation, LOD, and Sensitivity are log10-transformed.
- Hill coefficient n is analyzed without transformation.
- Omnibus test: scipy.stats.f_oneway
- Post-hoc: statsmodels.stats.multicomp.pairwise_tukeyhsd
- Tukey HSD is run only when the omnibus ANOVA P < 0.05.

Usage:
    python sensor_attribute_anova_tukey.py input.xlsx
    python sensor_attribute_anova_tukey.py input.csv
    python sensor_attribute_anova_tukey.py input.xlsx --sheet Attributes
    python sensor_attribute_anova_tukey.py input.xlsx --output-prefix HSL_sensor_stats
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


PROMOTERS = ["PFtsH", "PBla", "PVeg"]

ATTRIBUTES = [
    "Kd",
    "n",
    "ymin",
    "ymax",
    "Fold activation",
    "LOD",
    "Sensitivity",
]

LOG10_ATTRIBUTES = {
    "Kd",
    "ymin",
    "ymax",
    "Fold activation",
    "LOD",
    "Sensitivity",
}


def read_input(path: Path, sheet_name=None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else 0)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def validate_columns(df: pd.DataFrame) -> None:
    required = {"Host", "HSL", "QSR", "Promoter", *ATTRIBUTES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))


def transform_values(values: np.ndarray, attribute: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if attribute in LOG10_ATTRIBUTES:
        if np.any(values <= 0):
            raise ValueError(
                f"{attribute} contains zero/negative values, so log10 transformation is invalid."
            )
        return np.log10(values)

    return values


def run_anova(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for (host, hsl, qsr), subset in df.groupby(["Host", "HSL", "QSR"], sort=False):
        for attribute in ATTRIBUTES:
            groups = []
            group_ns = {}

            for promoter in PROMOTERS:
                values = (
                    subset.loc[subset["Promoter"] == promoter, attribute]
                    .dropna()
                    .astype(float)
                    .to_numpy()
                )
                group_ns[promoter] = len(values)
                groups.append(transform_values(values, attribute))

            if any(len(g) < 2 for g in groups):
                results.append({
                    "Host": host,
                    "HSL": hsl,
                    "QSR": qsr,
                    "Attribute": attribute,
                    "Transform": "log10" if attribute in LOG10_ATTRIBUTES else "none",
                    "F_statistic": np.nan,
                    "df1": len(PROMOTERS) - 1,
                    "df2": sum(len(g) for g in groups) - len(PROMOTERS),
                    "P_value": np.nan,
                    "n_PFtsH": group_ns["PFtsH"],
                    "n_PBla": group_ns["PBla"],
                    "n_PVeg": group_ns["PVeg"],
                    "Significant_P_lt_0.05": False,
                    "Note": "Insufficient observations in one or more promoter groups",
                })
                continue

            f_stat, p_value = f_oneway(*groups)
            df1 = len(PROMOTERS) - 1
            df2 = sum(len(g) for g in groups) - len(PROMOTERS)

            results.append({
                "Host": host,
                "HSL": hsl,
                "QSR": qsr,
                "Attribute": attribute,
                "Transform": "log10" if attribute in LOG10_ATTRIBUTES else "none",
                "F_statistic": f_stat,
                "df1": df1,
                "df2": df2,
                "P_value": p_value,
                "n_PFtsH": group_ns["PFtsH"],
                "n_PBla": group_ns["PBla"],
                "n_PVeg": group_ns["PVeg"],
                "Significant_P_lt_0.05": bool(p_value < 0.05),
                "Note": "",
            })

    return pd.DataFrame(results)


def run_tukey(df: pd.DataFrame, anova_results: pd.DataFrame) -> pd.DataFrame:
    results = []

    significant = anova_results[
        anova_results["P_value"].notna() &
        (anova_results["P_value"] < 0.05)
    ]

    for _, anova_row in significant.iterrows():
        host = anova_row["Host"]
        hsl = anova_row["HSL"]
        qsr = anova_row["QSR"]
        attribute = anova_row["Attribute"]

        temp = df[
            (df["Host"] == host) &
            (df["HSL"] == hsl) &
            (df["QSR"] == qsr) &
            (df["Promoter"].isin(PROMOTERS))
        ][["Promoter", attribute]].dropna().copy()

        raw_values = temp[attribute].astype(float).to_numpy()

        if attribute in LOG10_ATTRIBUTES:
            if np.any(raw_values <= 0):
                raise ValueError(
                    f"{attribute} contains zero/negative values for "
                    f"{host} / {hsl} / {qsr}; cannot run log10 Tukey HSD."
                )
            temp["Analysis_value"] = np.log10(raw_values)
            transform = "log10"
        else:
            temp["Analysis_value"] = raw_values
            transform = "none"

        tukey = pairwise_tukeyhsd(
            endog=temp["Analysis_value"],
            groups=temp["Promoter"],
            alpha=0.05,
        )

        for row in tukey.summary().data[1:]:
            results.append({
                "Host": host,
                "HSL": hsl,
                "QSR": qsr,
                "Attribute": attribute,
                "Transform": transform,
                "Promoter_1": row[0],
                "Promoter_2": row[1],
                "Mean_difference_analysis_scale": row[2],
                "Tukey_adjusted_P": row[3],
                "Lower_95_CI": row[4],
                "Upper_95_CI": row[5],
                "Significant_Tukey_P_lt_0.05": bool(row[6]),
            })

    return pd.DataFrame(results)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for (host, hsl, qsr, promoter), subset in df.groupby(
        ["Host", "HSL", "QSR", "Promoter"], sort=False
    ):
        if promoter not in PROMOTERS:
            continue

        for attribute in ATTRIBUTES:
            values = subset[attribute].dropna().astype(float)
            if len(values) == 0:
                continue

            records.append({
                "Host": host,
                "HSL": hsl,
                "QSR": qsr,
                "Promoter": promoter,
                "Attribute": attribute,
                "n": len(values),
                "Mean": values.mean(),
                "SD": values.std(ddof=1) if len(values) > 1 else np.nan,
                "Minimum": values.min(),
                "Maximum": values.max(),
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Run one-way ANOVA and Tukey HSD on HSL sensor fitted attributes."
    )
    parser.add_argument("input_file", type=Path, help="Input .xlsx/.xls/.csv file")
    parser.add_argument(
        "--sheet",
        default=None,
        help="Excel sheet name. If omitted, the first sheet is used.",
    )
    parser.add_argument(
        "--output-prefix",
        default="sensor_attribute_stats",
        help="Prefix for output files",
    )
    args = parser.parse_args()

    df = read_input(args.input_file, args.sheet)
    validate_columns(df)

    anova_results = run_anova(df)
    tukey_results = run_tukey(df, anova_results)
    descriptive = descriptive_statistics(df)

    xlsx_out = Path(f"{args.output_prefix}.xlsx")
    csv_anova = Path(f"{args.output_prefix}_anova.csv")
    csv_tukey = Path(f"{args.output_prefix}_tukey.csv")
    csv_desc = Path(f"{args.output_prefix}_descriptive.csv")

    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        anova_results.to_excel(writer, sheet_name="ANOVA", index=False)
        tukey_results.to_excel(writer, sheet_name="Tukey_HSD", index=False)
        descriptive.to_excel(writer, sheet_name="Descriptive", index=False)

    anova_results.to_csv(csv_anova, index=False)
    tukey_results.to_csv(csv_tukey, index=False)
    descriptive.to_csv(csv_desc, index=False)

    print(f"Saved: {xlsx_out}")
    print(f"Saved: {csv_anova}")
    print(f"Saved: {csv_tukey}")
    print(f"Saved: {csv_desc}")


if __name__ == "__main__":
    main()
