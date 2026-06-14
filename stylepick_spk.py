import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── PAGE CONFIG ───
st.set_page_config(
    page_title="StylePick SPK",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f8f9fa; }
[data-testid="stSidebar"] { background: #1e1b4b; }
[data-testid="stSidebar"] * { color: #e0e7ff !important; }
[data-testid="stSidebar"] .stRadio label { color: #e0e7ff !important; font-weight: 600; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #fff !important; }
.metric-card {
    background: white; border-radius: 12px; padding: 16px;
    border: 1px solid #e5e7eb; text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.metric-num { font-size: 28px; font-weight: 700; color: #4338ca; }
.metric-lbl { font-size: 12px; color: #9ca3af; margin-top: 4px; }
.rank-1 { background: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 13px; }
.rank-2 { background: #f3f4f6; color: #4b5563; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 13px; }
.rank-3 { background: #fee2e2; color: #991b1b; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 13px; }
.best-box { background: linear-gradient(135deg,#4338ca,#3730a3); color: white; border-radius: 14px; padding: 20px; text-align: center; }
.best-box-topsis { background: linear-gradient(135deg,#059669,#047857); color: white; border-radius: 14px; padding: 20px; text-align: center; }
.conclusion-box { background: #ecfdf5; border-left: 4px solid #059669; border-radius: 0 10px 10px 0; padding: 16px; margin: 10px 0; }
.conclusion-diff { background: #fefce8; border-left: 4px solid #f59e0b; border-radius: 0 10px 10px 0; padding: 16px; margin: 10px 0; }
.formula-box { background: #eef2ff; border-left: 4px solid #4338ca; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; font-family: monospace; color: #3730a3; }
.step-box { background: white; border-radius: 10px; border: 1px solid #e5e7eb; padding: 14px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───
if 'alt_names' not in st.session_state:
    st.session_state.alt_names = ['Hoodie', 'Kaos', 'Kemeja']
if 'alt_vals' not in st.session_state:
    st.session_state.alt_vals = [
        [150000, 90, 85, 20],
        [80000,  85, 80, 10],
        [120000, 88, 90, 15],
    ]
if 'crit_labels' not in st.session_state:
    st.session_state.crit_labels = ['Harga', 'Rating', 'Kualitas', 'Diskon']
if 'crit_types' not in st.session_state:
    st.session_state.crit_types = ['cost', 'benefit', 'benefit', 'benefit']
if 'crit_weights' not in st.session_state:
    st.session_state.crit_weights = [0.30, 0.25, 0.25, 0.20]

# ─── HELPERS ───
def r4(x): return round(float(x), 4)
def f4(x): return f"{x:.4f}".replace('.', ',')
def frp(x): return f"Rp {int(x):,}".replace(',', '.')

def get_den():
    """Penyebut normalisasi vektor √(Σx²) — dipakai untuk TOPSIS."""
    vals = np.array(st.session_state.alt_vals, dtype=float)
    return np.sqrt((vals ** 2).sum(axis=0))

def get_R():
    """Normalisasi vektor r_ij = x_ij / √(Σx²_kj) — dipakai untuk TOPSIS."""
    den = get_den()
    vals = np.array(st.session_state.alt_vals, dtype=float)
    den_safe = np.where(den == 0, 1, den)
    return vals / den_safe

def get_maxmin():
    """Nilai max & min tiap kriteria — dipakai untuk normalisasi SAW."""
    vals = np.array(st.session_state.alt_vals, dtype=float)
    return vals.max(axis=0), vals.min(axis=0)

def get_R_saw():
    """
    Normalisasi SAW sesuai teori standar (Fishburn):
    - Benefit : r_ij = x_ij / max(x_j)
    - Cost    : r_ij = min(x_j) / x_ij
    """
    vals = np.array(st.session_state.alt_vals, dtype=float)
    vmax, vmin = get_maxmin()
    types = st.session_state.crit_types
    R = np.zeros_like(vals)
    for j, t in enumerate(types):
        if t == 'benefit':
            denom = vmax[j] if vmax[j] != 0 else 1
            R[:, j] = vals[:, j] / denom
        else:  # cost
            x_safe = np.where(vals[:, j] == 0, 1, vals[:, j])
            R[:, j] = vmin[j] / x_safe
    return R

def dense_rank(values, decimals=10):
    """
    Dense ranking: nilai yang identik (setelah dibulatkan) mendapat rank yang sama,
    rank berikutnya lanjut tanpa 'lompat' nomor.
    Contoh: [0.85, 0.85, 0.70] -> rank [1, 1, 2]
    """
    rv = np.round(np.asarray(values, dtype=float), decimals)
    uniq = np.unique(rv)[::-1]  # urut dari terbesar
    rank_map = {val: i+1 for i, val in enumerate(uniq)}
    return np.array([rank_map[v] for v in rv])

def calc_saw():
    """SAW: normalisasi max/min (benefit & cost) lalu Vi = Σ(Wj × r_ij)."""
    R = get_R_saw()
    W = np.array(st.session_state.crit_weights)
    Vi = (R * W).sum(axis=1)
    ranks = dense_rank(Vi)
    return Vi, R, ranks

def calc_topsis():
    R = get_R()
    W = np.array(st.session_state.crit_weights)
    V = R * W
    types = st.session_state.crit_types
    Ap = np.array([V[:,j].max() if types[j]=='benefit' else V[:,j].min() for j in range(len(types))])
    Am = np.array([V[:,j].min() if types[j]=='benefit' else V[:,j].max() for j in range(len(types))])
    Dp = np.sqrt(((V - Ap)**2).sum(axis=1))
    Dm = np.sqrt(((V - Am)**2).sum(axis=1))
    Ci = np.where((Dp+Dm)==0, 0, Dm/(Dp+Dm))
    ranks = dense_rank(Ci)
    return Ci, V, Ap, Am, Dp, Dm, ranks

def stars(v, mx):
    if mx == 0: return '☆☆☆☆☆'
    n = round(v/mx * 5)
    return '★'*n + '☆'*(5-n)

def verdict(v, mx, t):
    if mx == 0: return ('—', '#6b7280')
    p = v/mx
    thr = [0.9,0.75,0.6,0.45] if t=='saw' else [0.75,0.55,0.4,0.25]
    if p >= thr[0]: return ('⭐ Sangat Baik', '#166534')
    if p >= thr[1]: return ('👍 Baik',        '#065f46')
    if p >= thr[2]: return ('➖ Cukup',       '#92400e')
    if p >= thr[3]: return ('👎 Kurang',      '#991b1b')
    return ('👎 Sangat Kurang', '#7f1d1d')

ALT_COLORS = ['#4338ca','#059669','#d97706','#dc2626','#7c3aed']

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("## 👗 StylePick SPK")
    st.markdown("---")
    menu = st.radio("Navigasi", [
        "📊 Data & Kriteria",
        "⚖️ Bobot Kriteria",
        "⚡ Metode SAW",
        "🎯 Metode TOPSIS",
        "🏆 Hasil Akhir"
    ])
    st.markdown("---")
    total_w = round(sum(st.session_state.crit_weights), 2)
    bobot_valid = abs(total_w - 1.0) < 0.005
    if bobot_valid:
        st.success(f"✅ Total bobot: **{total_w:.2f}**")
    else:
        st.error(f"⚠️ Total bobot: **{total_w:.2f}** (harus 1.00)")
    st.markdown("---")
    st.markdown("**Metode:**")
    st.markdown("- SAW (Simple Additive Weighting)")
    st.markdown("- TOPSIS")
    st.markdown("**Normalisasi:**")
    st.markdown("- SAW: Max/Min (benefit/cost)")
    st.markdown("- TOPSIS: Vektor")

# Hentikan perhitungan jika total bobot belum valid (kecuali di tab Data & Bobot)
if menu in ["⚡ Metode SAW", "🎯 Metode TOPSIS", "🏆 Hasil Akhir"] and not bobot_valid:
    st.title(menu)
    st.error(f"⚠️ Total bobot kriteria saat ini **{total_w:.2f}**, harus tepat **1.00** sebelum perhitungan bisa ditampilkan.")
    st.info("Silakan buka tab **⚖️ Bobot Kriteria** dan atur ulang slider bobot sampai totalnya 1.00.")
    st.stop()

# ════════════════════════════════════
#  TAB 1: DATA & KRITERIA
# ════════════════════════════════════
if menu == "📊 Data & Kriteria":
    st.title("📊 Data Alternatif & Kriteria")

    # Stats
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{len(st.session_state.alt_names)}</div><div class="metric-lbl">Alternatif Baju</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{len(st.session_state.crit_labels)}</div><div class="metric-lbl">Kriteria</div></div>', unsafe_allow_html=True)
    with c3:
        tw = round(sum(st.session_state.crit_weights),2)
        color = "#059669" if abs(tw-1)<0.005 else "#dc2626"
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:{color}">{tw:.2f}</div><div class="metric-lbl">Total Bobot</div></div>', unsafe_allow_html=True)
    with c4:
        nc = sum(1 for t in st.session_state.crit_types if t=='cost')
        st.markdown(f'<div class="metric-card"><div class="metric-num">{nc}</div><div class="metric-lbl">Kriteria Cost</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Edit Alternatif
    st.subheader("🧥 Edit Data Alternatif Baju")
    st.caption("Klik sel untuk mengedit langsung")

    df_alt = pd.DataFrame(
        st.session_state.alt_vals,
        columns=st.session_state.crit_labels,
        index=st.session_state.alt_names
    )
    df_alt.index.name = "Baju"

    edited = st.data_editor(
        df_alt,
        use_container_width=True,
        num_rows="dynamic",
        key="alt_editor"
    )

    if st.button("💾 Simpan Perubahan Data", type="primary"):
        st.session_state.alt_names = list(edited.index)
        st.session_state.alt_vals  = edited.values.tolist()
        st.success("✅ Data berhasil disimpan!")
        st.rerun()

    st.markdown("---")

    # Edit Kriteria
    st.subheader("⚙️ Edit Kriteria Penilaian")

    df_crit = pd.DataFrame({
        'Nama Kriteria': st.session_state.crit_labels,
        'Tipe': st.session_state.crit_types,
        'Bobot': st.session_state.crit_weights
    })

    edited_crit = st.data_editor(
        df_crit,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Tipe": st.column_config.SelectboxColumn(options=["benefit","cost"]),
            "Bobot": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        },
        key="crit_editor"
    )

    if st.button("💾 Simpan Perubahan Kriteria", type="primary"):
        old_labels = st.session_state.crit_labels
        new_labels = edited_crit['Nama Kriteria'].tolist()
        new_types  = edited_crit['Tipe'].tolist()
        new_weights= edited_crit['Bobot'].fillna(0).tolist()

        old_vals = np.array(st.session_state.alt_vals, dtype=float)
        n_alt = old_vals.shape[0]
        new_vals = np.zeros((n_alt, len(new_labels)))

        # Untuk kolom kriteria yang masih ada (cocokkan berdasarkan urutan/nama lama),
        # salin nilai lama. Kolom baru diisi 0 dan bisa diedit di tabel alternatif.
        for new_j, lbl in enumerate(new_labels):
            if new_j < len(old_labels) and lbl == old_labels[new_j]:
                # posisi sama & nama sama -> kolom lama
                new_vals[:, new_j] = old_vals[:, new_j]
            elif lbl in old_labels:
                old_j = old_labels.index(lbl)
                new_vals[:, new_j] = old_vals[:, old_j]
            elif new_j < old_vals.shape[1]:
                # kriteria baru di posisi kolom lama (mis. ganti nama) -> salin posisi
                new_vals[:, new_j] = old_vals[:, new_j]
            # else: kriteria baru -> tetap 0

        st.session_state.crit_labels  = new_labels
        st.session_state.crit_types   = new_types
        st.session_state.crit_weights = new_weights
        st.session_state.alt_vals     = new_vals.tolist()

        st.success("✅ Kriteria berhasil disimpan! Kolom baru (jika ada) berisi 0 — silakan isi nilainya di tabel Data Alternatif Baju.")
        st.rerun()

    st.markdown("---")

    # Preview Matriks
    st.subheader("📋 Preview Matriks Keputusan")
    df_prev = pd.DataFrame(
        st.session_state.alt_vals,
        columns=st.session_state.crit_labels,
        index=st.session_state.alt_names
    )
    # Format per kolom: hanya "Harga" → Rp, lainnya → integer
    col_fmt = {}
    for col in st.session_state.crit_labels:
        if col.lower() == 'harga':
            col_fmt[col] = lambda x: f"Rp {int(x):,}".replace(',', '.')
        else:
            col_fmt[col] = lambda x: str(int(x))
    styled = df_prev.style.format(col_fmt).set_properties(**{'text-align': 'center'})
    st.dataframe(styled, use_container_width=True)

# ════════════════════════════════════
#  TAB 2: BOBOT
# ════════════════════════════════════
elif menu == "⚖️ Bobot Kriteria":
    st.title("⚖️ Atur Bobot Kriteria")
    st.markdown("Geser slider untuk mengubah bobot. **Total bobot harus tepat 1.00.**")

    st.markdown("---")
    new_weights = []
    cols = st.columns(len(st.session_state.crit_labels))
    for i, (lbl, tp, w) in enumerate(zip(
        st.session_state.crit_labels,
        st.session_state.crit_types,
        st.session_state.crit_weights
    )):
        with cols[i]:
            badge = "🟢 Benefit" if tp=='benefit' else "🔴 Cost"
            st.caption(badge)
            nw = st.slider(lbl, 0.0, 0.6, float(w), 0.01, key=f"w_{i}")
            new_weights.append(nw)
            st.markdown(f"**{nw:.2f}** ({int(nw*100)}%)")

    st.session_state.crit_weights = new_weights
    total = round(sum(new_weights), 2)

    st.markdown("---")
    if abs(total - 1.0) < 0.005:
        st.success(f"✅ Total bobot: **{total:.2f}** — Sudah benar!")
    else:
        st.error(f"⚠️ Total bobot: **{total:.2f}** — Harus tepat 1.00!")

    st.markdown("---")
    st.subheader("📊 Grafik Bobot Real-time")

    c1, c2 = st.columns(2)

    with c1:
        # Donut
        fig_donut = go.Figure(go.Pie(
            labels=[f"{l} ({int(w*100)}%)" for l,w in zip(st.session_state.crit_labels, new_weights)],
            values=[int(w*100) for w in new_weights],
            hole=0.55,
            marker_colors=ALT_COLORS[:len(new_weights)],
            textinfo='percent+label',
            textfont_size=11
        ))
        fig_donut.update_layout(
            title="Proporsi Bobot (%)",
            showlegend=True,
            legend=dict(orientation='h', y=-0.15),
            margin=dict(t=40,b=60,l=0,r=0),
            height=320,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        # Stacked bar kontribusi
        R = get_R_saw()
        fig_stack = go.Figure()
        colors_stack = ['#4338ca','#f59e0b','#059669','#dc2626','#7c3aed']
        for j, (lbl, w) in enumerate(zip(st.session_state.crit_labels, new_weights)):
            kontrib = [round(w * R[i][j], 4) for i in range(len(st.session_state.alt_names))]
            fig_stack.add_trace(go.Bar(
                name=lbl,
                x=st.session_state.alt_names,
                y=kontrib,
                marker_color=colors_stack[j % len(colors_stack)],
                text=[f"{v:.4f}".replace('.',',') for v in kontrib],
                textposition='inside',
                textfont_size=10
            ))
        fig_stack.update_layout(
            barmode='stack', title="Kontribusi W×r per Kriteria",
            xaxis_title="Alternatif", yaxis_title="Nilai",
            yaxis=dict(range=[0,1]),
            legend=dict(orientation='h', y=-0.25),
            margin=dict(t=40,b=80,l=0,r=0),
            height=320,
            paper_bgcolor='white', plot_bgcolor='white'
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    # Tabel bobot
    st.subheader("📋 Tabel Bobot Kriteria")
    df_w = pd.DataFrame({
        'Kriteria': st.session_state.crit_labels,
        'Tipe': st.session_state.crit_types,
        'Bobot': [f"{w:.2f}" for w in new_weights],
        'Persentase': [f"{int(w*100)}%" for w in new_weights],
        'Keterangan': ['Semakin kecil semakin baik' if t=='cost' else 'Semakin besar semakin baik' for t in st.session_state.crit_types]
    })
    st.dataframe(df_w, use_container_width=True, hide_index=True)

# ════════════════════════════════════
#  TAB 3: SAW
# ════════════════════════════════════
elif menu == "⚡ Metode SAW":
    st.title("⚡ Metode SAW (Simple Additive Weighting)")

    st.markdown('<div class="formula-box">Normalisasi SAW (sesuai teori):<br>• Kriteria <b>Benefit</b>: r_ij = x_ij / max(x_j)<br>• Kriteria <b>Cost</b>: r_ij = min(x_j) / x_ij<br><br>Langkah 2 — Nilai preferensi: Vi = Σ (Wj × r_ij)  ← terbesar = terbaik</div>', unsafe_allow_html=True)

    Vi, R, ranks = calc_saw()
    W = st.session_state.crit_weights
    names = st.session_state.alt_names
    labels = st.session_state.crit_labels
    types = st.session_state.crit_types
    vmax, vmin = get_maxmin()
    maxVi = Vi.max()

    # Langkah 1: Nilai Max & Min tiap kriteria
    st.subheader("🔢 Nilai Max & Min Tiap Kriteria")
    df_mm = pd.DataFrame(
        [st.session_state.alt_vals[i] for i in range(len(names))] + [vmax.tolist(), vmin.tolist()],
        columns=labels,
        index=names + ['Max', 'Min']
    )
    def fmt_mm(col):
        def _fmt(x):
            if col.lower() == 'harga':
                return f"Rp {int(x):,}".replace(',', '.')
            if x == int(x):
                return str(int(x))
            return f"{x:.4f}"
        return _fmt
    mm_fmt = {col: fmt_mm(col) for col in labels}
    st.dataframe(df_mm.style.format(mm_fmt), use_container_width=True)

    # Contoh manual — ambil contoh kriteria benefit & cost pertama
    benefit_idx = next((j for j,t in enumerate(types) if t=='benefit'), 0)
    cost_idx    = next((j for j,t in enumerate(types) if t=='cost'), None)

    ex_b_name = names[0]; ex_b_lbl = labels[benefit_idx]
    ex_b_x = st.session_state.alt_vals[0][benefit_idx]
    ex_b_max = vmax[benefit_idx]
    ex_b_r = ex_b_x / ex_b_max if ex_b_max else 0
    st.info(f"📌 **Contoh Benefit — {ex_b_name}, {ex_b_lbl}:** r = x ÷ max = {ex_b_x:g} ÷ {ex_b_max:g} = **{ex_b_r:.4f}**")

    if cost_idx is not None:
        ex_c_name = names[0]; ex_c_lbl = labels[cost_idx]
        ex_c_x = st.session_state.alt_vals[0][cost_idx]
        ex_c_min = vmin[cost_idx]
        ex_c_r = ex_c_min / ex_c_x if ex_c_x else 0
        st.info(f"📌 **Contoh Cost — {ex_c_name}, {ex_c_lbl}:** r = min ÷ x = {ex_c_min:g} ÷ {ex_c_x:g} = **{ex_c_r:.4f}**")

    # Langkah 2: Normalisasi
    st.subheader("📐 Langkah 1 — Matriks Normalisasi (r_ij)")
    df_R = pd.DataFrame(R, columns=labels, index=names)
    df_R.index.name = "Jenis Baju"
    st.dataframe(df_R.style.format("{:.4f}"), use_container_width=True)

    # Langkah 3: Vi
    st.subheader("➕ Langkah 2 — Perhitungan Nilai Vi")
    rows_vi = []
    for i, name in enumerate(names):
        row = {}
        for j, lbl in enumerate(labels):
            row[f"{W[j]:.2f}×r({lbl[0]})"] = round(W[j]*R[i,j], 4)
        row['Vi'] = round(Vi[i], 4)
        row['Ranking'] = int(ranks[i])
        row['Bintang'] = stars(Vi[i], maxVi)
        vr, vc = verdict(Vi[i], maxVi, 'saw')
        row['Predikat'] = vr
        rows_vi.append(row)
    df_vi = pd.DataFrame(rows_vi, index=names)
    df_vi.index.name = "Jenis Baju"
    st.dataframe(df_vi, use_container_width=True)

    # Contoh Vi
    ex_n = names[ranks.argmin()]  # rank 1
    ex_i = list(ranks).index(1)
    ex_vi_parts = [f"({W[j]:.2f} × {R[ex_i,j]:.4f})" for j in range(len(labels))]
    st.info(f"📌 **Contoh Vi — {names[ex_i]}:** {' + '.join(ex_vi_parts)} = **{Vi[ex_i]:.4f}**")

    # Ranking
    st.subheader("🏅 Menghitung Nilai Akhir SAW (Ranking)")
    sorted_idx = np.argsort(Vi)[::-1]
    for pos, i in enumerate(sorted_idx):
        vr, vc = verdict(Vi[i], maxVi, 'saw')
        star = stars(Vi[i], maxVi)
        badge = "🥇 **Terbaik SAW**" if pos==0 else f"#{pos+1}"
        with st.container():
            col1, col2, col3, col4 = st.columns([0.5, 2, 2, 1])
            with col1:
                rank_labels = ["🥇","🥈","🥉"]
                st.markdown(f"### {rank_labels[pos] if pos<3 else str(pos+1)}")
            with col2:
                st.markdown(f"**{names[i]}**  {badge if pos==0 else ''}")
                st.progress(float(Vi[i]/maxVi) if maxVi else 0)
            with col3:
                st.markdown(f"{star}")
                st.markdown(f"<span style='color:{vc};font-weight:700'>{vr}</span>", unsafe_allow_html=True)
            with col4:
                st.metric("Vi", f4(Vi[i]))
        st.divider()

    # Grafik
    st.subheader("📊 Grafik Nilai Vi SAW")
    sorted_names = [names[i] for i in sorted_idx]
    sorted_vi    = [Vi[i] for i in sorted_idx]
    fig_saw = go.Figure(go.Bar(
        x=sorted_names, y=sorted_vi,
        marker_color=[ALT_COLORS[i % len(ALT_COLORS)] for i in sorted_idx],
        text=[f4(v) for v in sorted_vi],
        textposition='outside',
        textfont_size=12
    ))
    fig_saw.update_layout(
        title="Nilai Preferensi Vi — SAW",
        xaxis_title="Alternatif", yaxis_title="Nilai Vi",
        yaxis=dict(range=[0, max(sorted_vi)*1.2]),
        paper_bgcolor='white', plot_bgcolor='white',
        height=350, margin=dict(t=50,b=40)
    )
    fig_saw.add_hline(y=maxVi, line_dash='dash', line_color='#4338ca',
                      annotation_text=f"Tertinggi: {f4(maxVi)}")
    st.plotly_chart(fig_saw, use_container_width=True)

# ════════════════════════════════════
#  TAB 4: TOPSIS
# ════════════════════════════════════
elif menu == "🎯 Metode TOPSIS":
    st.title("🎯 Metode TOPSIS")

    st.markdown("""
    <div class="formula-box">
    Langkah 1 — r_ij = x_ij / √(Σ x²_kj)<br>
    Langkah 2 — v_ij = Wj × r_ij<br>
    Langkah 3 — A+: Benefit=max, Cost=min | A−: Benefit=min, Cost=max<br>
    Langkah 4 — D⁺ = √Σ(v_ij−A⁺)²  |  D⁻ = √Σ(v_ij−A⁻)²<br>
    Langkah 5 — Ci = D⁻ / (D⁺ + D⁻)  ← mendekati 1 = terbaik
    </div>
    """, unsafe_allow_html=True)

    Ci, V, Ap, Am, Dp, Dm, ranks = calc_topsis()
    names  = st.session_state.alt_names
    labels = st.session_state.crit_labels
    types  = st.session_state.crit_types
    W      = st.session_state.crit_weights
    maxCi  = Ci.max()

    # Langkah 1&2: v_ij
    st.subheader("📐 Langkah 1&2 — Matriks Normalisasi Terbobot (v_ij)")
    df_V = pd.DataFrame(V, columns=labels, index=names)
    df_V.index.name = "Jenis Baju"
    df_Ap = pd.DataFrame([Ap], columns=labels, index=['A+ (Ideal Positif)'])
    df_Am = pd.DataFrame([Am], columns=labels, index=['A− (Ideal Negatif)'])
    df_show = pd.concat([df_V, df_Ap, df_Am])
    st.dataframe(df_show.style.format("{:.4f}"), use_container_width=True)

    # Penjelasan A+ A-
    ap_info = " | ".join([f"**{lbl}** ({'min' if t=='cost' else 'max'}): {f4(Ap[j])}" for j,(lbl,t) in enumerate(zip(labels,types))])
    st.info(f"📌 **Penentuan A+ dan A−:** {ap_info}")

    # Langkah 3: D+
    st.subheader("📏 Langkah 3 — Jarak D⁺ ke Solusi Ideal Positif")
    rows_dp = {}
    for i, name in enumerate(names):
        row = {}
        for j, lbl in enumerate(labels):
            row[f"(v{lbl[0]}−A+)²"] = round((V[i,j]-Ap[j])**2, 6)
        row['D⁺'] = round(Dp[i], 4)
        rows_dp[name] = row
    df_dp = pd.DataFrame(rows_dp).T
    df_dp.index.name = "Jenis Baju"
    st.dataframe(df_dp.style.format("{:.4f}"), use_container_width=True)

    # Langkah 4: D-
    st.subheader("📏 Langkah 4 — Jarak D⁻ ke Solusi Ideal Negatif")
    rows_dm = {}
    for i, name in enumerate(names):
        row = {}
        for j, lbl in enumerate(labels):
            row[f"(v{lbl[0]}−A−)²"] = round((V[i,j]-Am[j])**2, 6)
        row['D⁻'] = round(Dm[i], 4)
        rows_dm[name] = row
    df_dm = pd.DataFrame(rows_dm).T
    df_dm.index.name = "Jenis Baju"
    st.dataframe(df_dm.style.format("{:.4f}"), use_container_width=True)

    # Langkah 5: Ci
    st.subheader("🏅 Langkah 5 — Nilai Preferensi Akhir (Ci) & Ranking TOPSIS")
    rows_ci = []
    sorted_idx = np.argsort(Ci)[::-1]
    for i, name in enumerate(names):
        vr, vc = verdict(Ci[i], maxCi, 'topsis')
        rows_ci.append({
            'Jenis Baju': name,
            'D⁺': f4(Dp[i]),
            'D⁻': f4(Dm[i]),
            'D⁺+D⁻': f4(round(Dp[i]+Dm[i],4)),
            'Ci': f4(Ci[i]),
            'Ranking': int(ranks[i]),
            'Bintang': stars(Ci[i], maxCi),
            'Predikat': vr
        })
    df_ci = pd.DataFrame(rows_ci)
    st.dataframe(df_ci, use_container_width=True, hide_index=True)

    # Contoh Ci
    best_i = sorted_idx[0]
    st.info(f"📌 **Contoh Ci — {names[best_i]}:** {f4(Dm[best_i])} ÷ ({f4(Dp[best_i])} + {f4(Dm[best_i])}) = {f4(Dm[best_i])} ÷ {f4(round(Dp[best_i]+Dm[best_i],4))} = **{f4(Ci[best_i])}**")

    # Ranking cards
    st.subheader("🏅 Ranking TOPSIS")
    for pos, i in enumerate(sorted_idx):
        vr, vc = verdict(Ci[i], maxCi, 'topsis')
        rank_labels_e = ["🥇","🥈","🥉"]
        col1, col2, col3, col4 = st.columns([0.5,2,2,1])
        with col1:
            st.markdown(f"### {rank_labels_e[pos] if pos<3 else str(pos+1)}")
        with col2:
            badge2 = "🥇 **Terbaik TOPSIS**" if pos==0 else ""
            st.markdown(f"**{names[i]}** {badge2}")
            st.progress(float(Ci[i]/maxCi) if maxCi else 0)
        with col3:
            st.markdown(stars(Ci[i],maxCi))
            st.markdown(f"<span style='color:{vc};font-weight:700'>{vr}</span>", unsafe_allow_html=True)
        with col4:
            st.metric("Ci", f4(Ci[i]))
        st.divider()

    # Grafik
    st.subheader("📊 Grafik TOPSIS")
    c1, c2 = st.columns(2)
    sorted_names_t = [names[i] for i in sorted_idx]
    sorted_ci      = [Ci[i]    for i in sorted_idx]
    sorted_dp      = [Dp[i]    for i in sorted_idx]
    sorted_dm      = [Dm[i]    for i in sorted_idx]

    with c1:
        fig_ci = go.Figure(go.Bar(
            x=sorted_names_t, y=sorted_ci,
            marker_color=[ALT_COLORS[i%len(ALT_COLORS)] for i in sorted_idx],
            text=[f4(v) for v in sorted_ci], textposition='outside'
        ))
        fig_ci.update_layout(title="Nilai Ci (Preferensi TOPSIS)", yaxis=dict(range=[0,1]),
            paper_bgcolor='white', plot_bgcolor='white', height=320, margin=dict(t=40,b=40))
        st.plotly_chart(fig_ci, use_container_width=True)

    with c2:
        fig_dp = go.Figure()
        fig_dp.add_trace(go.Bar(name='D⁺', x=sorted_names_t, y=sorted_dp,
            marker_color='rgba(220,38,38,.8)', text=[f4(v) for v in sorted_dp], textposition='outside'))
        fig_dp.add_trace(go.Bar(name='D⁻', x=sorted_names_t, y=sorted_dm,
            marker_color='rgba(5,150,105,.8)', text=[f4(v) for v in sorted_dm], textposition='outside'))
        fig_dp.update_layout(title="Jarak D⁺ vs D⁻", barmode='group',
            paper_bgcolor='white', plot_bgcolor='white', height=320, margin=dict(t=40,b=40),
            legend=dict(orientation='h',y=-0.2))
        st.plotly_chart(fig_dp, use_container_width=True)

# ════════════════════════════════════
#  TAB 5: HASIL AKHIR
# ════════════════════════════════════
elif menu == "🏆 Hasil Akhir":
    st.title("🏆 Hasil Akhir & Perbandingan")

    Vi, R, ranks_saw   = calc_saw()
    Ci, V, Ap, Am, Dp, Dm, ranks_top = calc_topsis()
    names  = st.session_state.alt_names
    labels = st.session_state.crit_labels
    maxVi  = Vi.max(); maxCi = Ci.max()

    best_saw_i   = np.argmax(Vi)
    best_topsis_i= np.argmax(Ci)
    agree = names[best_saw_i] == names[best_topsis_i]

    # Stats
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{len(names)}</div><div class="metric-lbl">Alternatif</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{len(labels)}</div><div class="metric-lbl">Kriteria</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#4338ca">{f4(maxVi)}</div><div class="metric-lbl">Vi Terbaik SAW</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#059669">{f4(maxCi)}</div><div class="metric-lbl">Ci Terbaik TOPSIS</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Radar Chart
    st.subheader("📡 Grafik Radar — Perbandingan Nilai Normalisasi")
    fig_radar = go.Figure()
    for i, name in enumerate(names):
        vals_r = [R[i,j] for j in range(len(labels))] + [R[i,0]]
        lbs_r  = labels + [labels[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_r, theta=lbs_r, fill='toself', name=name,
            line_color=ALT_COLORS[i%len(ALT_COLORS)],
            opacity=0.3
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True, legend=dict(orientation='h', y=-0.15),
        paper_bgcolor='white', height=420, margin=dict(t=30,b=80)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # Rekomendasi
    st.subheader("🥇 Rekomendasi Terbaik")
    col_saw, col_top = st.columns(2)

    with col_saw:
        vr_s, vc_s = verdict(Vi[best_saw_i], maxVi, 'saw')
        st.markdown(f"""
        <div class="best-box">
            <div style="font-size:12px;opacity:.8;margin-bottom:8px">⚡ Metode SAW</div>
            <div style="font-size:26px;font-weight:700;margin-bottom:8px">{names[best_saw_i]}</div>
            <div style="font-size:22px;margin-bottom:8px">{stars(Vi[best_saw_i],maxVi)}</div>
            <div style="font-size:14px;font-weight:700;background:rgba(255,255,255,.2);border-radius:20px;padding:4px 12px;display:inline-block">{vr_s}</div>
            <div style="font-size:13px;margin-top:10px;opacity:.8;font-family:monospace">Vi = {f4(Vi[best_saw_i])}</div>
            <div style="font-size:11px;opacity:.7;margin-top:4px">Ranking {int(ranks_saw[best_saw_i])} dari {len(names)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_top:
        vr_t, vc_t = verdict(Ci[best_topsis_i], maxCi, 'topsis')
        st.markdown(f"""
        <div class="best-box-topsis">
            <div style="font-size:12px;opacity:.8;margin-bottom:8px">🎯 Metode TOPSIS</div>
            <div style="font-size:26px;font-weight:700;margin-bottom:8px">{names[best_topsis_i]}</div>
            <div style="font-size:22px;margin-bottom:8px">{stars(Ci[best_topsis_i],maxCi)}</div>
            <div style="font-size:14px;font-weight:700;background:rgba(255,255,255,.2);border-radius:20px;padding:4px 12px;display:inline-block">{vr_t}</div>
            <div style="font-size:13px;margin-top:10px;opacity:.8;font-family:monospace">Ci = {f4(Ci[best_topsis_i])}</div>
            <div style="font-size:11px;opacity:.7;margin-top:4px">Ranking {int(ranks_top[best_topsis_i])} dari {len(names)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if agree:
        st.markdown(f"""
        <div class="conclusion-box">
        <b>✅ Kedua metode sepakat!</b><br>
        SAW dan TOPSIS sama-sama menyimpulkan bahwa <b>{names[best_saw_i]}</b> adalah pilihan baju terbaik dengan predikat <b>{vr_s}</b>. Konsistensi hasil dari dua metode berbeda memperkuat validitas keputusan ini.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="conclusion-diff">
        <b>⚠️ Hasil kedua metode berbeda</b><br>
        SAW merekomendasikan <b>{names[best_saw_i]}</b> (Vi={f4(Vi[best_saw_i])}), sedangkan TOPSIS merekomendasikan <b>{names[best_topsis_i]}</b> (Ci={f4(Ci[best_topsis_i])}). Perbedaan ini wajar karena TOPSIS mempertimbangkan jarak ke solusi ideal positif <i>dan</i> negatif secara bersamaan, sedangkan SAW hanya menjumlahkan nilai berbobot.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Ranking + Vi vs Ci chart
    st.subheader("📊 Grafik Perbandingan Vi SAW vs Ci TOPSIS")
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name='Vi (SAW)', x=names,
        y=[Vi[i] for i in range(len(names))],
        marker_color='rgba(67,56,202,.8)',
        text=[f4(Vi[i]) for i in range(len(names))], textposition='outside'
    ))
    fig_cmp.add_trace(go.Bar(
        name='Ci (TOPSIS)', x=names,
        y=[Ci[i] for i in range(len(names))],
        marker_color='rgba(5,150,105,.8)',
        text=[f4(Ci[i]) for i in range(len(names))], textposition='outside'
    ))
    fig_cmp.update_layout(
        barmode='group', yaxis=dict(range=[0,1]),
        paper_bgcolor='white', plot_bgcolor='white',
        height=350, margin=dict(t=30,b=40),
        legend=dict(orientation='h', y=-0.2)
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Tabel perbandingan
    st.subheader("🔄 Tabel Perbandingan Lengkap SAW vs TOPSIS")
    rows_cmp = []
    for i, name in enumerate(names):
        vrs, _ = verdict(Vi[i], maxVi, 'saw')
        vrt, _ = verdict(Ci[i], maxCi, 'topsis')
        df_diff = abs(int(ranks_saw[i]) - int(ranks_top[i]))
        rows_cmp.append({
            'Jenis Baju'   : name,
            'Rank SAW'     : int(ranks_saw[i]),
            'Vi'           : f4(Vi[i]),
            'Bintang SAW'  : stars(Vi[i], maxVi),
            'Predikat SAW' : vrs,
            'Rank TOPSIS'  : int(ranks_top[i]),
            'Ci'           : f4(Ci[i]),
            'Bintang TOPSIS': stars(Ci[i], maxCi),
            'Predikat TOPSIS': vrt,
            'Selisih Rank' : 'Sama' if df_diff==0 else f'+{df_diff}'
        })
    df_cmp = pd.DataFrame(rows_cmp)
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    # Kesimpulan akademik
    st.markdown("---")
    st.subheader("📝 Kesimpulan & Analisis Akademik")
    st.markdown(f"""
    **Metode SAW (Simple Additive Weighting):**
    Menggunakan normalisasi berdasarkan nilai maksimum dan minimum (max/min normalization): kriteria **Benefit** dinormalisasi dengan rumus r_ij = x_ij / max(x_j), sedangkan kriteria **Cost** dinormalisasi dengan rumus r_ij = min(x_j) / x_ij. Nilai ternormalisasi kemudian dikalikan bobot masing-masing kriteria dan dijumlahkan menjadi nilai preferensi Vi. Alternatif dengan **nilai Vi terbesar** merupakan pilihan terbaik. Metode ini sederhana, transparan, dan mudah diverifikasi secara manual.

    **Metode TOPSIS (Technique for Order Preference by Similarity to Ideal Solution):**
    Menggunakan normalisasi vektor (r_ij = x_ij / akar(Sigma x_kj kuadrat)) kemudian dikalikan bobot menjadi matriks terbobot v_ij. Selanjutnya ditentukan Solusi Ideal Positif (A+) sebagai kondisi terbaik dan Solusi Ideal Negatif (A-) sebagai kondisi terburuk. Dihitung jarak D+ ke A+ dan D- ke A- untuk tiap alternatif. Nilai preferensi Ci = D-/(D++D-) mendekati 1 berarti alternatif mendekati kondisi ideal terbaik. Metode ini lebih komprehensif karena mempertimbangkan dua sisi sekaligus.

    **Perbandingan SAW vs TOPSIS:**
    SAW cocok untuk keputusan yang memerlukan transparansi dan kemudahan verifikasi karena normalisasi dan perhitungannya sederhana. TOPSIS lebih robust untuk data dengan variasi signifikan antar kriteria karena mempertimbangkan jarak ke kondisi ideal dan non-ideal secara bersamaan.

    **Kesimpulan Akhir:**
    {"Kedua metode SAW dan TOPSIS **sepakat** bahwa **" + names[best_saw_i] + "** merupakan pilihan baju terbaik. Konsistensi hasil dari dua pendekatan berbeda memperkuat validitas keputusan ini." if agree else f"SAW merekomendasikan **{names[best_saw_i]}** (Vi={f4(Vi[best_saw_i])}) dan TOPSIS merekomendasikan **{names[best_topsis_i]}** (Ci={f4(Ci[best_topsis_i])}). Perbedaan ini wajar karena SAW menggunakan normalisasi max/min sedangkan TOPSIS menggunakan normalisasi vektor, sehingga skala perbandingan antar alternatif berbeda."}
    """)
