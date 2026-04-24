import pandas as pd


def iqr_outlier_stats(series: pd.Series) -> tuple[int, float]:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return 0, 0.0

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    outliers_n = int(((series < lim_inf) | (series > lim_sup)).sum())
    outliers_pct = outliers_n / len(series)
    return outliers_n, outliers_pct
