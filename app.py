
import streamlit as st
import os
from inference_pipeline import load_artifacts, predict_klasifikasi

st.set_page_config(
    page_title="Klasifikasi Perkara Pidana",
    page_icon="⚖️",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .hero-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #a0aec0;
        margin-top: 0.5rem;
    }
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #60a5fa;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }
    .result-box-khusus {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-box-umum {
        background: linear-gradient(135deg, #0c1a3a, #1e3a5f);
        border-left: 5px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .result-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 0.2rem;
    }
    .footer-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
    }
    .footer-title {
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .footer-text {
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.8;
        margin-top: 0.5rem;
    }
    .footer-link {
        color: #60a5fa;
        text-decoration: none;
    }
    .badge {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.78rem;
        color: #94a3b8;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <div class="hero-title">⚖️ Klasifikasi Perkara Pidana</div>
    <div class="hero-subtitle">
        Prediksi otomatis klasifikasi perkara pidana Indonesia<br>
        menggunakan model machine learning berbasis data putusan pengadilan
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stat Cards ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""<div class="stat-card">
        <div class="stat-number">94%</div>
        <div class="stat-label">Akurasi</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="stat-card">
        <div class="stat-number">93.6%</div>
        <div class="stat-label">F1-Score</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="stat-card">
        <div class="stat-number">97.9%</div>
        <div class="stat-label">ROC-AUC</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""<div class="stat-card">
        <div class="stat-number">16K+</div>
        <div class="stat-label">Data Latih</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Panduan ──────────────────────────────────────────────
with st.expander("Panduan — Baca sebelum menggunakan", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Pidana Khusus", "Pidana Umum", "Tentang Model"])

    with tab1:
        st.markdown("""
        #### Apa itu Pidana Khusus?
        Tindak pidana yang diatur **di luar KUHP** melalui undang-undang tersendiri.
        Umumnya menyangkut kepentingan negara, masyarakat luas, atau kejahatan terorganisir.

        | Jenis Perkara | Dasar Hukum |
        |---|---|
        | Narkotika | UU No. 35 Tahun 2009 |
        | Korupsi | UU No. 31/1999 jo. UU No. 20/2001 |
        | Pencucian Uang | UU No. 8 Tahun 2010 |
        | ITE | UU No. 11/2008 jo. UU No. 19/2016 |
        | KDRT | UU No. 23 Tahun 2004 |
        | Perlindungan Anak | UU No. 35 Tahun 2014 |
        | Terorisme | UU No. 5 Tahun 2018 |

        **Ciri khas:** Ancaman hukuman lebih berat, ditangani unit jaksa khusus (KPK, BNN, dll).
        """)

    with tab2:
        st.markdown("""
        #### Apa itu Pidana Umum?
        Tindak pidana yang diatur dalam **KUHP** (Kitab Undang-Undang Hukum Pidana).
        Mencakup kejahatan sehari-hari yang umum terjadi di masyarakat.

        | Jenis Perkara | Pasal KUHP |
        |---|---|
        | Pencurian | Pasal 362 - 367 |
        | Penganiayaan | Pasal 351 - 358 |
        | Penipuan | Pasal 378 - 395 |
        | Pembunuhan | Pasal 338 - 350 |
        | Pemerasan | Pasal 368 - 371 |
        | Penggelapan | Pasal 372 - 377 |
        | Pemerkosaan | Pasal 285 - 288 |

        **Ciri khas:** Ditangani pengadilan negeri biasa, jaksa dari kejaksaan negeri setempat.
        """)

    with tab3:
        st.markdown("""
        #### Cara Kerja Model
        Model membaca pasal dakwaan, data terdakwa, dan pertimbangan hakim,
        lalu memprediksi apakah perkara tersebut termasuk pidana khusus atau pidana umum.

        #### Algoritma
        **LightGBM** — model gradient boosting yang dilatih pada 16.414 putusan pengadilan
        dari 5 provinsi (Jatim, Jateng, Jabar, DKI Jakarta, DIY).

        #### Keterbatasan
        - Paling akurat untuk perkara dari **5 provinsi utama** di atas
        - Untuk provinsi lain, akurasi bisa lebih rendah
        - **Bukan pengganti analisis hukum profesional**
        """)

st.divider()

# ── Form ─────────────────────────────────────────────────
st.subheader("Informasi Perkara")

JENIS_PERKARA_MAP = {
    "Pidana Khusus (Narkotika, Korupsi, ITE, dll)": "pid.sus",
    "Pidana Umum (Pencurian, Penganiayaan, Penipuan, dll)": "pid.b",
}
KELOMPOK_UMUR_MAP = {
    "Remaja (di bawah 18 tahun)": "remaja",
    "Dewasa Muda (18–25 tahun)": "dewasa-muda",
    "Dewasa (26–40 tahun)": "dewasa",
    "Paruh Baya (41–55 tahun)": "paruh-baya",
    "Senior (56–65 tahun)": "senior",
    "Lansia (di atas 65 tahun)": "lansia",
    "Tidak Diketahui": "tidak-diketahui",
}
PROVINSI_MAP = {
    "Jawa Timur": "jatim",
    "Jawa Tengah": "jateng",
    "Jawa Barat": "jabar",
    "DKI Jakarta": "jkt",
    "DI Yogyakarta": "yogya",
    "Provinsi Lainnya": "lainnya",
}

col1, col2 = st.columns(2)

with col1:
    jenis_label   = st.selectbox("Jenis Perkara", list(JENIS_PERKARA_MAP.keys()))
    umur_label    = st.selectbox("Kelompok Umur Terdakwa", list(KELOMPOK_UMUR_MAP.keys()))
    umur          = st.number_input("Umur Terdakwa (tahun)", min_value=0, max_value=100, value=25,
                                     help="Isi 0 jika umur tidak diketahui")
    provinsi_label = st.selectbox("Provinsi Pengadilan", list(PROVINSI_MAP.keys()))

with col2:
    pasal           = st.text_area("Pasal Dakwaan", height=100,
                                    placeholder="Contoh: Pasal 114 dan 112 UU No.35 Tahun 2009 tentang Narkotika")
    hal_meringankan = st.text_area("Hal yang Meringankan Terdakwa", height=100,
                                    placeholder="Contoh: bersikap sopan, menyesali perbuatan, belum pernah dihukum")
    hal_memberatkan = st.text_area("Hal yang Memberatkan Terdakwa", height=100,
                                    placeholder="Contoh: meresahkan masyarakat, tidak kooperatif")

st.markdown("<br>", unsafe_allow_html=True)

# ── Prediksi ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return load_artifacts(base_dir)

artifacts = load_model()

if st.button("🔍  Prediksi Klasifikasi", type="primary", use_container_width=True):
    if not pasal.strip():
        st.warning("Pasal dakwaan tidak boleh kosong.")
    else:
        with st.spinner("Memproses prediksi..."):
            hasil = predict_klasifikasi(
                artifacts,
                JENIS_PERKARA_MAP[jenis_label],
                KELOMPOK_UMUR_MAP[umur_label],
                float(umur),
                PROVINSI_MAP[provinsi_label],
                pasal, hal_meringankan, hal_memberatkan
            )

        label_tampil = "Pidana Khusus" if hasil.prediksi == "pidana-khusus" else "Pidana Umum"
        box_class    = "result-box-khusus" if hasil.prediksi == "pidana-khusus" else "result-box-umum"
        icon         = "🔴" if hasil.prediksi == "pidana-khusus" else "🔵"

        st.markdown(f"""
        <div class="{box_class}">
            <div class="result-label">Hasil Klasifikasi</div>
            <div class="result-value">{icon} {label_tampil}</div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        col_a.metric("Tingkat Keyakinan", f"{hasil.confidence * 100:.1f}%")
        lawan = [v for k, v in hasil.probabilitas.items() if k != hasil.prediksi][0]
        col_b.metric("Kemungkinan Kelas Lain", f"{lawan * 100:.1f}%")

        st.progress(hasil.confidence)

        with st.expander("Lihat Detail Probabilitas"):
            for kelas, prob in hasil.probabilitas.items():
                label_k = "Pidana Khusus" if kelas == "pidana-khusus" else "Pidana Umum"
                st.write(f"{label_k}: {prob * 100:.2f}%")

        st.caption("Hasil prediksi bersifat indikatif dan tidak menggantikan penilaian hukum resmi.")

# ── Footer ───────────────────────────────────────────────
st.markdown("""
<div class="footer-box">
    <div class="footer-title">⚖️ Klasifikasi Perkara Pidana Indonesia</div>
    <div class="footer-text">
        Sistem prediksi klasifikasi perkara pidana berbasis machine learning.<br>
        Dibangun menggunakan algoritma LightGBM yang dilatih pada data putusan pengadilan Indonesia.<br><br>
        <span class="badge">Dibuat oleh Janferi</span>
        <span class="badge">Juni 2026</span>
        <span class="badge">LightGBM</span>
        <span class="badge">Python</span><br><br>
        <strong style="color:#94a3b8">Sumber Data:</strong><br>
        <a class="footer-link" href="https://github.com/ir-nlp-csui/indo-law" target="_blank">
            Indo-Law Dataset — ir-nlp-csui/indo-law
        </a><br>
        Dataset putusan pengadilan Indonesia oleh tim NLP,
        Fakultas Ilmu Komputer, Universitas Indonesia.<br><br>
         <a class="footer-link" href="mailto:esekiel241@gmail.com">esekiel241@gmail.com</a>
        &nbsp;·&nbsp;
         <a class="footer-link" href="https://github.com/Janferi" target="_blank">github.com/Janferi</a><br><br>
        <em style="color:#475569">
            Hasil prediksi bersifat indikatif dan tidak menggantikan keputusan atau penilaian hukum resmi.
        </em>
    </div>
</div>
""", unsafe_allow_html=True)
