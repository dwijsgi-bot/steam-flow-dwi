#!/usr/bin/env python3
"""
Analisis data flow steam dari Excel.

File  : JFE LOGGER 2026_10_30.xlsx  |  Sheet: editing2
Kolom (dibaca berdasarkan posisi, lalu di-relabel):
    kolom 1 -> Tanggal
    kolom 2 -> Time
    kolom 3 -> Flow Steam (Kg/h)
    kolom 4 -> Totalizer (Kg)

Cara pakai (jalankan dari folder tempat file Excel berada):
    python analisis_steam_flow.py
    python analisis_steam_flow.py "JFE LOGGER 2026_10_30.xlsx" --sheet editing2

Output (folder 'hasil_analisis'):
    - ringkasan_analisis.xlsx  (statistik, profil jam, harian, anomali, dll.)
    - grafik_*.png             (time series, profil beban, konsumsi harian, LDC, histogram)

Dependensi: pandas, numpy, matplotlib, openpyxl
    pip install pandas numpy matplotlib openpyxl
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# KONFIGURASI DEFAULT (bisa diubah lewat argumen CLI)
# ----------------------------------------------------------------------
DEFAULT_FILE = "JFE LOGGER 2026_10_30.xlsx"   # file Excel default
DEFAULT_SHEET = "editing2"                      # sheet yang dianalisis

# Label kolom baku (kolom 1-4 di sheet akan diganti dengan label ini)
LABEL_TANGGAL = "Tanggal"
LABEL_WAKTU = "Time"
LABEL_FLOW = "Flow Steam (Kg/h)"
LABEL_TOT = "Totalizer (Kg)"

FLOW_UNIT = "Kg/h"       # satuan flow sesaat
TOT_UNIT = "Kg"          # satuan totalizer
# Faktor konversi integral flow (per jam) -> satuan totalizer.
# Kg/h -> Kg = 1 (flow Kg/h dikali jam = Kg)
FLOW_TO_TOT_FACTOR = 1.0
MAX_GAP_MENIT = 15       # selisih waktu > ini dianggap data hilang (gap)
OUTLIER_IQR_K = 3.0      # pengali IQR untuk deteksi outlier flow
OUTPUT_DIR = "hasil_analisis"


# ----------------------------------------------------------------------
# 1. LOAD & CLEAN
# ----------------------------------------------------------------------
def cari_kolom(df, kunci_list, hindari=()):
    """Cari nama kolom yang mengandung salah satu kata kunci."""
    for col in df.columns:
        nama = str(col).lower().strip()
        if any(k in nama for k in kunci_list) and not any(h in nama for h in hindari):
            return col
    return None


def deteksi_kolom(df):
    c_tgl = cari_kolom(df, ["tanggal", "tgl", "date"])
    c_wkt = cari_kolom(df, ["waktu", "jam", "time"], hindari=["date"])
    c_tot = cari_kolom(df, ["totalizer", "total", "tot"])
    c_flow = cari_kolom(df, ["flow", "aliran"], hindari=["totalizer", "total", "tot"])

    # fallback: pakai urutan kolom 1-4
    if None in (c_tgl, c_wkt, c_flow, c_tot) and df.shape[1] >= 4:
        print("[!] Nama kolom tidak dikenali, memakai urutan kolom: "
              "tanggal, waktu, flow, totalizer")
        c_tgl, c_wkt, c_flow, c_tot = df.columns[:4]
    if None in (c_tgl, c_wkt, c_flow, c_tot):
        sys.exit("ERROR: Tidak dapat menentukan 4 kolom (tanggal, waktu, flow, totalizer).")
    return c_tgl, c_wkt, c_flow, c_tot


def gabung_datetime(tgl, wkt):
    """Gabungkan kolom tanggal + waktu (mendukung time, string, atau pecahan Excel)."""
    tgl = pd.to_datetime(tgl, errors="coerce", dayfirst=True).dt.normalize()

    if pd.api.types.is_numeric_dtype(wkt):          # pecahan hari Excel (0.5 = 12:00)
        delta = pd.to_timedelta(wkt.astype(float), unit="D")
    elif pd.api.types.is_datetime64_any_dtype(wkt):
        delta = pd.to_timedelta(wkt.dt.strftime("%H:%M:%S"), errors="coerce")
    else:                                            # datetime.time / string
        delta = pd.to_timedelta(wkt.astype(str), errors="coerce")
    return tgl + delta


def load_data(path, sheet=0):
    raw = pd.read_excel(path, sheet_name=sheet)
    if raw.shape[1] < 4:
        sys.exit(f"ERROR: Sheet '{sheet}' hanya punya {raw.shape[1]} kolom, butuh minimal 4.")

    # Kolom dibaca berdasarkan POSISI (1-4), lalu diberi label baru
    label_lama = list(raw.columns[:4])
    raw = raw.iloc[:, :4].copy()
    raw.columns = [LABEL_TANGGAL, LABEL_WAKTU, LABEL_FLOW, LABEL_TOT]
    print("Relabel kolom:")
    for lama, baru in zip(label_lama, raw.columns):
        print(f"   '{lama}'  ->  '{baru}'")

    df = pd.DataFrame({
        "timestamp": gabung_datetime(raw[LABEL_TANGGAL], raw[LABEL_WAKTU]),
        "flow": pd.to_numeric(raw[LABEL_FLOW], errors="coerce"),
        "totalizer": pd.to_numeric(raw[LABEL_TOT], errors="coerce"),
    })

    n_awal = len(df)
    df = df.dropna(subset=["timestamp"])
    n_ts_invalid = n_awal - len(df)
    n_dup = df.duplicated(subset="timestamp").sum()
    df = (df.sort_values("timestamp")
            .drop_duplicates(subset="timestamp", keep="first")
            .set_index("timestamp"))

    info = {
        "Jumlah baris awal": n_awal,
        "Baris timestamp invalid (dibuang)": n_ts_invalid,
        "Baris timestamp duplikat (dibuang)": int(n_dup),
        "Flow kosong/non-numerik": int(df["flow"].isna().sum()),
        "Totalizer kosong/non-numerik": int(df["totalizer"].isna().sum()),
        "Jumlah baris dipakai": len(df),
    }
    return df, info


# ----------------------------------------------------------------------
# 2. ANALISIS
# ----------------------------------------------------------------------
def tambah_kolom_turunan(df):
    df = df.copy()
    df["dt_jam"] = df.index.to_series().diff().dt.total_seconds() / 3600.0
    df["d_totalizer"] = df["totalizer"].diff()
    return df


def statistik_flow(df):
    f = df["flow"].dropna()
    pos = f[f > 0]
    idx_max, idx_min = f.idxmax(), f.idxmin()
    stat = {
        f"Jumlah data": len(f),
        f"Periode awal": df.index.min(),
        f"Periode akhir": df.index.max(),
        f"Rata-rata flow ({FLOW_UNIT})": f.mean(),
        f"Median flow ({FLOW_UNIT})": f.median(),
        f"Std deviasi ({FLOW_UNIT})": f.std(),
        f"Minimum ({FLOW_UNIT})": f.min(),
        f"Waktu minimum": idx_min,
        f"Maksimum / peak ({FLOW_UNIT})": f.max(),
        f"Waktu peak": idx_max,
        f"P5 ({FLOW_UNIT})": f.quantile(0.05),
        f"P50 ({FLOW_UNIT})": f.quantile(0.50),
        f"P95 ({FLOW_UNIT})": f.quantile(0.95),
        f"Rata-rata saat flow > 0 ({FLOW_UNIT})": pos.mean() if len(pos) else np.nan,
        "Load factor (rata-rata / peak)": f.mean() / f.max() if f.max() else np.nan,
        "Turndown (peak / P5)": f.max() / f.quantile(0.05) if f.quantile(0.05) > 0 else np.nan,
        "Persentase waktu flow = 0 (%)": (f <= 0).mean() * 100,
        "Koefisien variasi (std/mean)": f.std() / f.mean() if f.mean() else np.nan,
    }
    return pd.DataFrame({"Parameter": stat.keys(), "Nilai": stat.values()})


def analisis_totalizer(df):
    """Konsumsi dari totalizer (selisih), dengan penanganan reset/rollover."""
    d = df["d_totalizer"]
    anomali_neg = d < 0
    konsumsi_tot = d[~anomali_neg].sum(skipna=True)
    return konsumsi_tot, anomali_neg.sum()


def konsumsi_dari_flow(df):
    """Integral flow terhadap waktu (metode trapesium) dalam satuan totalizer."""
    d = df[["flow", "dt_jam"]].dropna().copy()
    # abaikan celah data yang terlalu besar agar integral tidak melenceng
    d = d[d["dt_jam"] <= MAX_GAP_MENIT / 60.0]
    flow_prev = df["flow"].shift(1).loc[d.index]
    energi = ((d["flow"] + flow_prev) / 2.0 * d["dt_jam"]).sum()
    return energi * FLOW_TO_TOT_FACTOR


def konsumsi_periode(df, freq):
    """Konsumsi per periode berdasarkan selisih totalizer (selisih negatif diabaikan)."""
    d = df["d_totalizer"].where(df["d_totalizer"] >= 0, 0.0).fillna(0.0)
    g = df.assign(konsumsi=d).resample(freq)
    out = pd.DataFrame({
        f"Konsumsi ({TOT_UNIT})": g["konsumsi"].sum(),
        f"Flow rata-rata ({FLOW_UNIT})": g["flow"].mean(),
        f"Flow maks ({FLOW_UNIT})": g["flow"].max(),
        f"Flow min ({FLOW_UNIT})": g["flow"].min(),
        "Jumlah data": g["flow"].count(),
    })
    out[f"Waktu peak"] = g["flow"].apply(lambda s: s.idxmax() if s.notna().any() else pd.NaT)
    return out


def profil_per_jam(df):
    g = df.groupby(df.index.hour)["flow"]
    out = pd.DataFrame({
        f"Rata-rata ({FLOW_UNIT})": g.mean(),
        f"Median ({FLOW_UNIT})": g.median(),
        f"Min ({FLOW_UNIT})": g.min(),
        f"Maks ({FLOW_UNIT})": g.max(),
        f"P90 ({FLOW_UNIT})": g.quantile(0.9),
    })
    out.index.name = "Jam"
    return out


def profil_hari_dalam_minggu(df):
    nama = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    g = df.groupby(df.index.dayofweek)["flow"]
    out = pd.DataFrame({f"Rata-rata ({FLOW_UNIT})": g.mean(),
                        f"Maks ({FLOW_UNIT})": g.max()})
    out.index = [nama[i] for i in out.index]
    out.index.name = "Hari"
    return out


def deteksi_gap(df):
    gap = df[df["dt_jam"] * 60 > MAX_GAP_MENIT]
    return pd.DataFrame({
        "Waktu setelah gap": gap.index,
        "Lama gap (menit)": (gap["dt_jam"] * 60).round(1).values,
    })


def deteksi_outlier(df):
    f = df["flow"].dropna()
    q1, q3 = f.quantile(0.25), f.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - OUTLIER_IQR_K * iqr, q3 + OUTLIER_IQR_K * iqr
    o = df[(df["flow"] < lo) | (df["flow"] > hi) | (df["flow"] < 0)]
    out = o[["flow"]].copy()
    out.index.name = "Waktu"
    out.attrs["batas"] = (lo, hi)
    return out


def deteksi_reset_totalizer(df):
    r = df[df["d_totalizer"] < 0]
    return pd.DataFrame({
        "Waktu": r.index,
        "Totalizer sebelumnya": (r["totalizer"] - r["d_totalizer"]).values,
        "Totalizer sesudahnya": r["totalizer"].values,
        f"Selisih ({TOT_UNIT})": r["d_totalizer"].values,
    })


def deteksi_flow_vs_totalizer_tidak_cocok(df, toleransi=0.25):
    """Tandai interval saat integral flow berbeda jauh dari kenaikan totalizer."""
    d = df.dropna(subset=["flow", "d_totalizer", "dt_jam"]).copy()
    d = d[(d["d_totalizer"] >= 0) & (d["dt_jam"] > 0) & (d["dt_jam"] <= MAX_GAP_MENIT / 60)]
    d["dari_flow"] = (d["flow"] + df["flow"].shift(1).loc[d.index]) / 2 * d["dt_jam"] * FLOW_TO_TOT_FACTOR
    d["selisih"] = d["d_totalizer"] - d["dari_flow"]
    ref = d["dari_flow"].abs().clip(lower=d["dari_flow"].abs().median() or 1e-9)
    d["rel_err"] = d["selisih"].abs() / ref
    bad = d[d["rel_err"] > toleransi]
    return bad[["flow", "d_totalizer", "dari_flow", "selisih"]].rename(columns={
        "flow": f"Flow ({FLOW_UNIT})",
        "d_totalizer": f"Δ Totalizer ({TOT_UNIT})",
        "dari_flow": f"Integral flow ({TOT_UNIT})",
        "selisih": f"Selisih ({TOT_UNIT})",
    })


def load_duration_curve(df):
    f = df["flow"].dropna().sort_values(ascending=False).reset_index(drop=True)
    pct = (f.index + 1) / len(f) * 100
    return pct, f


# ----------------------------------------------------------------------
# 3. GRAFIK
# ----------------------------------------------------------------------
def simpan(fig, nama):
    path = os.path.join(OUTPUT_DIR, nama)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def buat_grafik(df, harian, prof_jam):
    # a. Time series flow + totalizer
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(df.index, df["flow"], lw=0.7, color="tab:blue", label="Flow")
    ax1.set_ylabel(f"Flow ({FLOW_UNIT})", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["totalizer"], lw=0.9, color="tab:red", label="Totalizer")
    ax2.set_ylabel(f"Totalizer ({TOT_UNIT})", color="tab:red")
    ax1.set_title("Flow Steam & Totalizer")
    ax1.grid(alpha=0.3)
    simpan(fig, "grafik_1_timeseries.png")

    # b. Profil beban per jam
    fig, ax = plt.subplots(figsize=(10, 5))
    k = f"Rata-rata ({FLOW_UNIT})"
    ax.plot(prof_jam.index, prof_jam[k], marker="o", label="Rata-rata")
    ax.fill_between(prof_jam.index, prof_jam[f"Min ({FLOW_UNIT})"],
                    prof_jam[f"Maks ({FLOW_UNIT})"], alpha=0.15, label="Rentang min-maks")
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Jam"); ax.set_ylabel(f"Flow ({FLOW_UNIT})")
    ax.set_title("Profil Beban Steam per Jam"); ax.legend(); ax.grid(alpha=0.3)
    simpan(fig, "grafik_2_profil_jam.png")

    # c. Konsumsi harian
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(harian.index, harian[f"Konsumsi ({TOT_UNIT})"], width=0.8)
    ax.set_ylabel(f"Konsumsi ({TOT_UNIT})"); ax.set_title("Konsumsi Steam Harian")
    ax.grid(alpha=0.3, axis="y")
    fig.autofmt_xdate()
    simpan(fig, "grafik_3_konsumsi_harian.png")

    # d. Load duration curve
    pct, f = load_duration_curve(df)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pct, f)
    ax.set_xlabel("% waktu flow ≥ nilai"); ax.set_ylabel(f"Flow ({FLOW_UNIT})")
    ax.set_title("Load Duration Curve"); ax.grid(alpha=0.3)
    simpan(fig, "grafik_4_load_duration_curve.png")

    # e. Histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["flow"].dropna(), bins=50, edgecolor="white")
    ax.set_xlabel(f"Flow ({FLOW_UNIT})"); ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi Flow Steam"); ax.grid(alpha=0.3)
    simpan(fig, "grafik_5_histogram.png")


# ----------------------------------------------------------------------
# 4. EXPORT
# ----------------------------------------------------------------------
def export_excel(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for nama, d in sheets.items():
            d.to_excel(xw, sheet_name=nama[:31], index=not isinstance(d.index, pd.RangeIndex))
            ws = xw.sheets[nama[:31]]
            for col in ws.columns:
                lebar = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(lebar + 3, 45)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    global FLOW_UNIT, TOT_UNIT, FLOW_TO_TOT_FACTOR, MAX_GAP_MENIT, OUTPUT_DIR

    ap = argparse.ArgumentParser(description="Analisis data flow steam dari Excel")
    ap.add_argument("file", nargs="?", default=DEFAULT_FILE,
                    help=f"Path file Excel (default: {DEFAULT_FILE})")
    ap.add_argument("--sheet", default=DEFAULT_SHEET,
                    help=f"Nama/indeks sheet (default: {DEFAULT_SHEET})")
    ap.add_argument("--flow-unit", default=FLOW_UNIT)
    ap.add_argument("--tot-unit", default=TOT_UNIT)
    ap.add_argument("--factor", type=float, default=FLOW_TO_TOT_FACTOR,
                    help="Faktor konversi integral flow/jam ke satuan totalizer")
    ap.add_argument("--max-gap", type=float, default=MAX_GAP_MENIT,
                    help="Batas gap data (menit)")
    ap.add_argument("--out", default=OUTPUT_DIR)
    a = ap.parse_args()

    FLOW_UNIT, TOT_UNIT = a.flow_unit, a.tot_unit
    FLOW_TO_TOT_FACTOR, MAX_GAP_MENIT, OUTPUT_DIR = a.factor, a.max_gap, a.out
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sheet = int(a.sheet) if str(a.sheet).isdigit() else a.sheet
    df, info = load_data(a.file, sheet)
    df = tambah_kolom_turunan(df)

    # --- analisis ---
    stat = statistik_flow(df)
    kons_tot, n_reset = analisis_totalizer(df)
    kons_flow = konsumsi_dari_flow(df)
    selisih_pct = (kons_flow - kons_tot) / kons_tot * 100 if kons_tot else np.nan

    rekon = pd.DataFrame({
        "Parameter": [f"Konsumsi dari totalizer ({TOT_UNIT})",
                      f"Konsumsi dari integral flow ({TOT_UNIT})",
                      "Selisih flow vs totalizer (%)",
                      "Jumlah reset/rollover totalizer"],
        "Nilai": [kons_tot, kons_flow, selisih_pct, n_reset],
    })

    jam = konsumsi_periode(df, "h")
    harian = konsumsi_periode(df, "D")
    bulanan = konsumsi_periode(df, "MS")
    prof_jam = profil_per_jam(df)
    prof_hari = profil_hari_dalam_minggu(df)
    gap = deteksi_gap(df)
    outlier = deteksi_outlier(df)
    reset = deteksi_reset_totalizer(df)
    mismatch = deteksi_flow_vs_totalizer_tidak_cocok(df)
    kualitas = pd.DataFrame({"Parameter": info.keys(), "Nilai": info.values()})

    # --- grafik ---
    buat_grafik(df, harian, prof_jam)

    # --- export ---
    xlsx = os.path.join(OUTPUT_DIR, "ringkasan_analisis.xlsx")
    export_excel(xlsx, {
        "Kualitas Data": kualitas,
        "Statistik Flow": stat,
        "Rekonsiliasi": rekon,
        "Per Jam": jam,
        "Harian": harian,
        "Bulanan": bulanan,
        "Profil Jam": prof_jam,
        "Profil Hari": prof_hari,
        "Gap Data": gap,
        "Outlier Flow": outlier,
        "Reset Totalizer": reset,
        "Flow vs Totalizer": mismatch,
    })

    # --- ringkasan di terminal ---
    pd.set_option("display.width", 140)
    print("\n=== KUALITAS DATA ===");  print(kualitas.to_string(index=False))
    print("\n=== STATISTIK FLOW ===");  print(stat.to_string(index=False))
    print("\n=== REKONSILIASI KONSUMSI ===");  print(rekon.to_string(index=False))
    print(f"\nGap data (> {MAX_GAP_MENIT:.0f} menit): {len(gap)} kejadian")
    print(f"Outlier flow: {len(outlier)} titik")
    print(f"Reset/rollover totalizer: {len(reset)} kejadian")
    print(f"Interval flow vs totalizer tidak cocok: {len(mismatch)} titik")
    print(f"\nHasil tersimpan di folder: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
