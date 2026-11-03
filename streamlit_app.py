"""
Aplikasi Streamlit: Analisis Data Flow Steam (JFE LOGGER)

Kolom di sheet (dibaca berdasarkan POSISI, lalu di-relabel):
    kolom 1 -> Tanggal | kolom 2 -> Time | kolom 3 -> Flow Steam (Kg/h) | kolom 4 -> Totalizer (Kg)

Jalankan lokal :  streamlit run streamlit_app.py
Streamlit Cloud:  Main file path = streamlit_app.py  (wajib ada requirements.txt)
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = ["Tanggal", "Time", "Flow Steam (Kg/h)", "Totalizer (Kg)"]


# ----------------------------------------------------------------------
# LOAD & CLEAN
# ----------------------------------------------------------------------
def gabung_datetime(tgl, wkt):
    """Gabungkan kolom tanggal + waktu (time, teks, datetime, atau pecahan Excel)."""
    tgl = pd.to_datetime(tgl, errors="coerce", dayfirst=True).dt.normalize()
    if pd.api.types.is_numeric_dtype(wkt):
        delta = pd.to_timedelta(wkt.astype(float), unit="D")
    elif pd.api.types.is_datetime64_any_dtype(wkt):
        delta = pd.to_timedelta(wkt.dt.strftime("%H:%M:%S"), errors="coerce")
    else:
        delta = pd.to_timedelta(wkt.astype(str), errors="coerce")
    return tgl + delta


def siapkan_data(raw):
    """raw = DataFrame mentah dari Excel (minimal 4 kolom). Return (df, info)."""
    label_lama = [str(c) for c in raw.columns[:4]]
    r = raw.iloc[:, :4].copy()
    r.columns = LABELS

    df = pd.DataFrame({
        "timestamp": gabung_datetime(r["Tanggal"], r["Time"]),
        "flow": pd.to_numeric(r["Flow Steam (Kg/h)"], errors="coerce"),
        "totalizer": pd.to_numeric(r["Totalizer (Kg)"], errors="coerce"),
    })
    n_awal = len(df)
    df = df.dropna(subset=["timestamp"])
    n_invalid = n_awal - len(df)
    n_dup = int(df.duplicated(subset="timestamp").sum())
    df = (df.sort_values("timestamp")
            .drop_duplicates(subset="timestamp", keep="first")
            .set_index("timestamp"))
    df["dt_jam"] = df.index.to_series().diff().dt.total_seconds() / 3600.0
    df["d_totalizer"] = df["totalizer"].diff()

    info = pd.DataFrame({
        "Parameter": ["Jumlah baris awal", "Timestamp invalid (dibuang)",
                      "Timestamp duplikat (dibuang)", "Flow kosong/non-numerik",
                      "Totalizer kosong/non-numerik", "Jumlah baris dipakai"],
        "Nilai": [n_awal, n_invalid, n_dup, int(df["flow"].isna().sum()),
                  int(df["totalizer"].isna().sum()), len(df)],
    })
    relabel = pd.DataFrame({"Label lama": label_lama, "Label baru": LABELS})
    return df, info, relabel


# ----------------------------------------------------------------------
# ANALISIS
# ----------------------------------------------------------------------
def statistik_flow(df, fu):
    f = df["flow"].dropna()
    pos = f[f > 0]
    p5 = f.quantile(0.05)
    stat = {
        "Jumlah data": len(f),
        "Periode awal": df.index.min(),
        "Periode akhir": df.index.max(),
        f"Rata-rata ({fu})": f.mean(),
        f"Median ({fu})": f.median(),
        f"Std deviasi ({fu})": f.std(),
        f"Minimum ({fu})": f.min(),
        "Waktu minimum": f.idxmin(),
        f"Maksimum / peak ({fu})": f.max(),
        "Waktu peak": f.idxmax(),
        f"P5 ({fu})": p5,
        f"P50 ({fu})": f.quantile(0.5),
        f"P95 ({fu})": f.quantile(0.95),
        f"Rata-rata saat flow > 0 ({fu})": pos.mean() if len(pos) else np.nan,
        "Load factor (rata-rata / peak)": f.mean() / f.max() if f.max() else np.nan,
        "Turndown (peak / P5)": f.max() / p5 if p5 > 0 else np.nan,
        "Waktu flow <= 0 (%)": (f <= 0).mean() * 100,
        "Koefisien variasi (std/mean)": f.std() / f.mean() if f.mean() else np.nan,
    }
    out = pd.DataFrame({"Parameter": list(stat.keys()), "Nilai": list(stat.values())})
    return out


def konsumsi_tot(df):
    d = df["d_totalizer"]
    neg = d < 0
    return d[~neg].sum(skipna=True), int(neg.sum())


def konsumsi_flow(df, factor, max_gap_menit):
    d = df[["flow", "dt_jam"]].dropna()
    d = d[d["dt_jam"] <= max_gap_menit / 60.0]
    prev = df["flow"].shift(1).loc[d.index]
    return ((d["flow"] + prev) / 2.0 * d["dt_jam"]).sum() * factor


def konsumsi_periode(df, freq, fu, tu):
    d = df["d_totalizer"].where(df["d_totalizer"] >= 0, 0.0).fillna(0.0)
    g = df.assign(konsumsi=d).resample(freq)
    out = pd.DataFrame({
        f"Konsumsi ({tu})": g["konsumsi"].sum(),
        f"Flow rata-rata ({fu})": g["flow"].mean(),
        f"Flow maks ({fu})": g["flow"].max(),
        f"Flow min ({fu})": g["flow"].min(),
        "Jumlah data": g["flow"].count(),
    })
    out["Waktu peak"] = g["flow"].apply(lambda s: s.idxmax() if s.notna().any() else pd.NaT)
    return out


def profil_jam(df, fu):
    g = df.groupby(df.index.hour)["flow"]
    out = pd.DataFrame({f"Rata-rata ({fu})": g.mean(), f"Median ({fu})": g.median(),
                        f"Min ({fu})": g.min(), f"Maks ({fu})": g.max(),
                        f"P90 ({fu})": g.quantile(0.9)})
    out.index.name = "Jam"
    return out


def profil_hari(df, fu):
    nama = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    g = df.groupby(df.index.dayofweek)["flow"]
    out = pd.DataFrame({f"Rata-rata ({fu})": g.mean(), f"Maks ({fu})": g.max()})
    out.index = [nama[i] for i in out.index]
    out.index.name = "Hari"
    return out


def deteksi_gap(df, max_gap_menit):
    gap = df[df["dt_jam"] * 60 > max_gap_menit]
    return pd.DataFrame({"Waktu setelah gap": gap.index,
                         "Lama gap (menit)": (gap["dt_jam"] * 60).round(1).values})


def deteksi_outlier(df, k):
    f = df["flow"].dropna()
    q1, q3 = f.quantile(0.25), f.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    o = df[(df["flow"] < lo) | (df["flow"] > hi) | (df["flow"] < 0)][["flow"]].copy()
    o.index.name = "Waktu"
    return o, lo, hi


def deteksi_reset(df, tu):
    r = df[df["d_totalizer"] < 0]
    return pd.DataFrame({"Waktu": r.index,
                         "Totalizer sebelumnya": (r["totalizer"] - r["d_totalizer"]).values,
                         "Totalizer sesudahnya": r["totalizer"].values,
                         f"Selisih ({tu})": r["d_totalizer"].values})


def deteksi_mismatch(df, factor, max_gap_menit, fu, tu, toleransi=0.25):
    d = df.dropna(subset=["flow", "d_totalizer", "dt_jam"]).copy()
    d = d[(d["d_totalizer"] >= 0) & (d["dt_jam"] > 0) & (d["dt_jam"] <= max_gap_menit / 60)]
    if d.empty:
        return pd.DataFrame()
    d["dari_flow"] = (d["flow"] + df["flow"].shift(1).loc[d.index]) / 2 * d["dt_jam"] * factor
    d["selisih"] = d["d_totalizer"] - d["dari_flow"]
    med = d["dari_flow"].abs().median()
    ref = d["dari_flow"].abs().clip(lower=med if med and med > 0 else 1e-9)
    bad = d[(d["selisih"].abs() / ref) > toleransi]
    return bad[["flow", "d_totalizer", "dari_flow", "selisih"]].rename(columns={
        "flow": f"Flow ({fu})", "d_totalizer": f"Δ Totalizer ({tu})",
        "dari_flow": f"Integral flow ({tu})", "selisih": f"Selisih ({tu})"})


def batas_kontrol(df, sigma_ucl=3.0, sigma_uwl=2.0):
    """Control chart: CL = rata-rata; UWL/LWL = CL ± 2σ; UCL/LCL = CL ± 3σ."""
    f = df["flow"].dropna()
    mu, sd = f.mean(), f.std()
    return {"CL": mu, "sd": sd,
            "UCL": mu + sigma_ucl * sd, "UWL": mu + sigma_uwl * sd,
            "LWL": mu - sigma_uwl * sd, "LCL": mu - sigma_ucl * sd,
            "sigma_ucl": sigma_ucl, "sigma_uwl": sigma_uwl}


def tabel_batas(lim, fu):
    baris = [
        (f"UCL (CL + {lim['sigma_ucl']:g}σ)", lim["UCL"]),
        (f"UWL (CL + {lim['sigma_uwl']:g}σ)", lim["UWL"]),
        ("CL (rata-rata)", lim["CL"]),
        (f"LWL (CL - {lim['sigma_uwl']:g}σ)", lim["LWL"]),
        (f"LCL (CL - {lim['sigma_ucl']:g}σ)", lim["LCL"]),
        ("σ (std deviasi)", lim["sd"]),
    ]
    out = pd.DataFrame(baris, columns=["Batas", f"Nilai ({fu})"])
    out["Catatan"] = ["" if (v >= 0 or n.startswith("σ")) else "negatif (flow fisik tidak < 0)"
                      for n, v in baris]
    return out


def deteksi_di_luar_batas(df, lim, fu):
    f = df["flow"]
    status = np.select(
        [f > lim["UCL"], f < lim["LCL"], f > lim["UWL"], f < lim["LWL"]],
        ["> UCL", "< LCL", "> UWL (warning)", "< LWL (warning)"], default="")
    out = pd.DataFrame({f"Flow ({fu})": f, "Status": status}, index=df.index)
    out = out[out["Status"] != ""]
    out.index.name = "Waktu"
    return out


def ringkas_di_luar_batas(di_luar, n_total):
    urut = ["> UCL", "> UWL (warning)", "< LWL (warning)", "< LCL"]
    rows = [(k, int((di_luar["Status"] == k).sum())) for k in urut]
    out = pd.DataFrame(rows, columns=["Status", "Jumlah titik"])
    out["% dari data"] = (out["Jumlah titik"] / n_total * 100).round(3)
    return out



# ----------------------------------------------------------------------
# GRAFIK
# ----------------------------------------------------------------------
def fig_timeseries(df, fu, tu):
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.plot(df.index, df["flow"], lw=0.7, color="tab:blue")
    ax1.set_ylabel(f"Flow ({fu})", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["totalizer"], lw=0.9, color="tab:red")
    ax2.set_ylabel(f"Totalizer ({tu})", color="tab:red")
    ax1.set_title("Flow Steam & Totalizer")
    ax1.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_control(df, lim, fu):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["flow"], lw=0.7, color="tab:blue", label="Flow")
    garis = [("UCL", lim["UCL"], "tab:red", "-"), ("UWL", lim["UWL"], "tab:orange", "--"),
             ("CL", lim["CL"], "tab:green", "-"), ("LWL", lim["LWL"], "tab:orange", "--"),
             ("LCL", lim["LCL"], "tab:red", "-")]
    for nama, v, w, ls in garis:
        ax.axhline(v, color=w, ls=ls, lw=1.1, label=f"{nama} = {v:,.0f}")
    luar = df[(df["flow"] > lim["UCL"]) | (df["flow"] < lim["LCL"])]
    if len(luar):
        ax.scatter(luar.index, luar["flow"], color="red", s=18, zorder=5, label="Di luar UCL/LCL")
    ax.set_ylabel(f"Flow ({fu})"); ax.set_title("Control Chart Flow Steam")
    ax.legend(loc="upper right", fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.autofmt_xdate(); fig.tight_layout()
    return fig


def fig_profil_jam(p, fu):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(p.index, p[f"Rata-rata ({fu})"], marker="o", label="Rata-rata")
    ax.fill_between(p.index, p[f"Min ({fu})"], p[f"Maks ({fu})"], alpha=0.15,
                    label="Rentang min-maks")
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Jam"); ax.set_ylabel(f"Flow ({fu})")
    ax.set_title("Profil Beban per Jam"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_harian(h, tu):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(h.index, h[f"Konsumsi ({tu})"], width=0.8)
    ax.set_ylabel(f"Konsumsi ({tu})"); ax.set_title("Konsumsi Harian")
    ax.grid(alpha=0.3, axis="y")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_ldc(df, fu):
    f = df["flow"].dropna().sort_values(ascending=False).reset_index(drop=True)
    pct = (f.index + 1) / len(f) * 100
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(pct, f)
    ax.set_xlabel("% waktu flow ≥ nilai"); ax.set_ylabel(f"Flow ({fu})")
    ax.set_title("Load Duration Curve"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_hist(df, fu):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(df["flow"].dropna(), bins=50, edgecolor="white")
    ax.set_xlabel(f"Flow ({fu})"); ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi Flow"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# EXPORT
# ----------------------------------------------------------------------
def export_excel_bytes(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for nama, d in sheets.items():
            if d is None or len(d) == 0:
                d = pd.DataFrame({"Info": ["Tidak ada data"]})
            d = d.copy()
            for c in d.columns:                      # nilai campuran -> teks agar aman
                if d[c].dtype == object:
                    d[c] = d[c].astype(str)
            d.to_excel(xw, sheet_name=nama[:31], index=not isinstance(d.index, pd.RangeIndex))
            ws = xw.sheets[nama[:31]]
            for col in ws.columns:
                lebar = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(lebar + 3, 45)
    return buf.getvalue()


# ----------------------------------------------------------------------
# UI STREAMLIT
# ----------------------------------------------------------------------
def main():
    import streamlit as st

    st.set_page_config(page_title="Analisis Steam Flow", page_icon="♨️", layout="wide")
    st.title("♨️ Analisis Data Flow Steam")
    st.caption("Kolom dibaca berdasarkan posisi: 1 = Tanggal, 2 = Time, "
               "3 = Flow Steam (Kg/h), 4 = Totalizer (Kg)")

    @st.cache_data(show_spinner=False)
    def baca_sheet_names(data: bytes):
        return pd.ExcelFile(io.BytesIO(data)).sheet_names

    @st.cache_data(show_spinner="Membaca Excel...")
    def baca_excel(data: bytes, sheet):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet)

    with st.sidebar:
        st.header("Pengaturan")
        up = st.file_uploader("Upload file Excel", type=["xlsx", "xlsm", "xls"])
        fu = st.text_input("Satuan flow", "Kg/h")
        tu = st.text_input("Satuan totalizer", "Kg")
        factor = st.number_input("Faktor integral flow → totalizer", value=1.0, format="%.6f",
                                 help="Kg/h → Kg = 1. Kalau flow Kg/h tapi totalizer ton = 0.001")
        max_gap = st.number_input("Batas gap data (menit)", value=15.0, min_value=0.1)
        iqr_k = st.number_input("Pengali IQR outlier", value=3.0, min_value=0.5)
        sg_ucl = st.number_input("Sigma untuk UCL/LCL", value=3.0, min_value=0.5)
        sg_uwl = st.number_input("Sigma untuk UWL/LWL", value=2.0, min_value=0.5)

    if up is None:
        st.info("⬅️ Upload file Excel (mis. *JFE LOGGER 2026_10_30.xlsx*) di sidebar untuk mulai.")
        return

    data = up.getvalue()
    try:
        sheets = baca_sheet_names(data)
    except Exception as e:
        st.error(f"File Excel tidak bisa dibaca: {e}")
        return

    idx_def = sheets.index("editing2") if "editing2" in sheets else 0
    sheet = st.sidebar.selectbox("Sheet", sheets, index=idx_def)

    try:
        raw = baca_excel(data, sheet)
    except Exception as e:
        st.error(f"Gagal membaca sheet '{sheet}': {e}")
        return
    if raw.shape[1] < 4:
        st.error(f"Sheet '{sheet}' hanya punya {raw.shape[1]} kolom. Dibutuhkan minimal 4.")
        return

    try:
        df, info, relabel = siapkan_data(raw)
    except Exception as e:
        st.error(f"Gagal memproses data: {e}")
        st.dataframe(raw.head(20))
        return
    if len(df) < 3 or df["flow"].notna().sum() < 3:
        st.error("Data valid terlalu sedikit. Cek isi kolom Tanggal, Time, dan Flow.")
        st.dataframe(raw.head(20))
        return

    with st.expander("Relabel kolom & kualitas data", expanded=False):
        c1, c2 = st.columns(2)
        c1.dataframe(relabel, hide_index=True)
        c2.dataframe(info, hide_index=True)

    # ---- hitung ----
    stat = statistik_flow(df, fu)
    k_tot, n_reset = konsumsi_tot(df)
    k_flow = konsumsi_flow(df, factor, max_gap)
    selisih_pct = (k_flow - k_tot) / k_tot * 100 if k_tot else np.nan
    jam = konsumsi_periode(df, "60min", fu, tu)
    harian = konsumsi_periode(df, "D", fu, tu)
    bulanan = konsumsi_periode(df, "MS", fu, tu)
    pj, ph = profil_jam(df, fu), profil_hari(df, fu)
    gap = deteksi_gap(df, max_gap)
    outlier, lo, hi = deteksi_outlier(df, iqr_k)
    reset = deteksi_reset(df, tu)
    mism = deteksi_mismatch(df, factor, max_gap, fu, tu)
    lim = batas_kontrol(df, sg_ucl, sg_uwl)
    tbl_batas = tabel_batas(lim, fu)
    di_luar = deteksi_di_luar_batas(df, lim, fu)
    rk_luar = ringkas_di_luar_batas(di_luar, int(df["flow"].notna().sum()))
    rekon = pd.DataFrame({
        "Parameter": [f"Konsumsi dari totalizer ({tu})", f"Konsumsi dari integral flow ({tu})",
                      "Selisih flow vs totalizer (%)", "Jumlah reset/rollover totalizer"],
        "Nilai": [k_tot, k_flow, selisih_pct, n_reset]})

    # ---- metrik utama ----
    f = df["flow"]
    m = st.columns(5)
    m[0].metric(f"Rata-rata ({fu})", f"{f.mean():,.1f}")
    m[1].metric(f"Peak ({fu})", f"{f.max():,.1f}", help=str(f.idxmax()))
    m[2].metric(f"Minimum ({fu})", f"{f.min():,.1f}")
    m[3].metric(f"Konsumsi totalizer ({tu})", f"{k_tot:,.0f}")
    m[4].metric("Selisih flow vs totalizer", "-" if np.isnan(selisih_pct) else f"{selisih_pct:+.2f}%")

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(
        ["📈 Grafik", "📉 Control Chart", "📊 Statistik", "🔥 Konsumsi", "🕒 Profil",
         "⚠️ Anomali", "🗂️ Data"])

    with t1:
        st.pyplot(fig_timeseries(df, fu, tu))
        a, b = st.columns(2)
        a.pyplot(fig_ldc(df, fu))
        b.pyplot(fig_hist(df, fu))

    with t2:
        st.pyplot(fig_control(df, lim, fu))
        a, b = st.columns(2)
        a.subheader("Batas kontrol"); a.dataframe(tbl_batas, hide_index=True)
        b.subheader("Titik di luar batas"); b.dataframe(rk_luar, hide_index=True)
        st.caption("UCL/LCL = CL ± σ·k (default 3σ), UWL/LWL = CL ± σ·k (default 2σ). "
                   "Batas dihitung dari seluruh data yang di-upload.")
        with st.expander(f"Daftar titik di luar UWL/LWL ({len(di_luar)})"):
            st.dataframe(di_luar)

    with t3:
        st.dataframe(stat.astype({"Nilai": str}), hide_index=True)
        st.subheader("Rekonsiliasi konsumsi")
        st.dataframe(rekon, hide_index=True)

    with t4:
        st.pyplot(fig_harian(harian, tu))
        st.subheader("Harian"); st.dataframe(harian)
        st.subheader("Per jam"); st.dataframe(jam)
        st.subheader("Bulanan"); st.dataframe(bulanan)

    with t5:
        st.pyplot(fig_profil_jam(pj, fu))
        a, b = st.columns(2)
        a.subheader("Per jam"); a.dataframe(pj)
        b.subheader("Per hari"); b.dataframe(ph)

    with t6:
        st.write(f"**Gap data** (> {max_gap:g} menit): {len(gap)} kejadian")
        st.dataframe(gap, hide_index=True)
        st.write(f"**Outlier flow** (di luar {lo:,.1f} – {hi:,.1f} atau negatif): {len(outlier)} titik")
        st.dataframe(outlier)
        st.write(f"**Reset/rollover totalizer**: {len(reset)} kejadian")
        st.dataframe(reset, hide_index=True)
        st.write(f"**Interval flow vs totalizer tidak cocok** (> 25%): {len(mism)} titik")
        st.dataframe(mism)

    with t7:
        st.dataframe(df.drop(columns=["dt_jam"]).rename(
            columns={"flow": f"Flow ({fu})", "totalizer": f"Totalizer ({tu})",
                     "d_totalizer": f"Δ Totalizer ({tu})"}))

    # ---- download ----
    xlsx = export_excel_bytes({
        "Kualitas Data": info, "Statistik Flow": stat.astype({"Nilai": str}),
        "Rekonsiliasi": rekon, "Per Jam": jam, "Harian": harian, "Bulanan": bulanan,
        "Profil Jam": pj, "Profil Hari": ph, "Gap Data": gap, "Outlier Flow": outlier,
        "Reset Totalizer": reset, "Flow vs Totalizer": mism,
        "Batas Kontrol": tbl_batas, "Di Luar Batas": di_luar})
    st.sidebar.download_button("⬇️ Download hasil (Excel)", xlsx,
                               file_name="ringkasan_analisis_steam.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
