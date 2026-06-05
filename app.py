
import streamlit as st
import joblib
import os
from inference_pipeline import load_artifacts, predict_klasifikasi

st.set_page_config(
    page_title="Klasifikasi Perkara Pidana",
    page_icon="⚖️",
    layout="centered"
)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return load_artifacts(base_dir)

artifacts = load_model()

JENIS_PERKARA_MAP = {
    "Pidana Khusus (Narkotika, Korupsi, ITE, dll)": "pid.sus",
    "Pidana Umum (Pencurian, Penganiayaan, Penipuan, dll)": "pid.b",
}

KELOMPOK_UMUR_MAP = {
    "Remaja (di bawah 18 tahun)": "remaja",
    "Dewasa Muda (18-25 tahun)": "dewasa-muda",
    "Dewasa (26-40 tahun)": "dewasa",
    "Paruh Baya (41-55 tahun)": "paruh-baya",
    "Senior (56-65 tahun)": "senior",
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

# Header
st.title("⚖️ Klasifikasi Perkara Pidana")
st.caption(
    "Sistem prediksi berbasis machine learning untuk mengklasifikasikan "
    "perkara pidana ke dalam kategori pidana khusus atau pidana umum."
)

st.divider()

# Form Input
st.subheader("Informasi Perkara")

col1, col2 = st.columns(2)

with col1:
    jenis_label = st.selectbox(
        "Jenis Perkara",
        list(JENIS_PERKARA_MAP.keys()),
        help="Pilih berdasarkan jenis pelanggaran yang didakwakan"
    )
    umur_label = st.selectbox(
        "Kelompok Umur Terdakwa",
        list(KELOMPOK_UMUR_MAP.keys())
    )
    umur = st.number_input(
        "Umur Terdakwa (tahun)",
        min_value=0, max_value=100, value=25,
        help="Isi 0 jika umur tidak diketahui"
    )
    provinsi_label = st.selectbox(
        "Provinsi Pengadilan",
        list(PROVINSI_MAP.keys()),
        help="Lokasi pengadilan negeri yang menangani perkara"
    )

with col2:
    pasal = st.text_area(
        "Pasal Dakwaan",
        height=100,
        placeholder="Contoh: Pasal 114 dan 112 UU No.35 Tahun 2009 tentang Narkotika"
    )
    hal_meringankan = st.text_area(
        "Hal yang Meringankan Terdakwa",
        height=100,
        placeholder="Contoh: bersikap sopan, menyesali perbuatan, belum pernah dihukum"
    )
    hal_memberatkan = st.text_area(
        "Hal yang Memberatkan Terdakwa",
        height=100,
        placeholder="Contoh: meresahkan masyarakat, tidak kooperatif"
    )

st.divider()

if st.button("Prediksi Klasifikasi", type="primary", use_container_width=True):
    if not pasal.strip():
        st.warning("Pasal dakwaan tidak boleh kosong.")
    else:
        jenis_perkara = JENIS_PERKARA_MAP[jenis_label]
        kelompok_umur = KELOMPOK_UMUR_MAP[umur_label]
        provinsi      = PROVINSI_MAP[provinsi_label]

        with st.spinner("Memproses prediksi..."):
            hasil = predict_klasifikasi(
                artifacts, jenis_perkara, kelompok_umur,
                float(umur), provinsi, pasal,
                hal_meringankan, hal_memberatkan
            )

        st.subheader("Hasil Prediksi")

        label_tampil = (
            "Pidana Khusus" if hasil.prediksi == "pidana-khusus"
            else "Pidana Umum"
        )

        if hasil.prediksi == "pidana-khusus":
            st.error(f"Klasifikasi: **{label_tampil}**")
        else:
            st.info(f"Klasifikasi: **{label_tampil}**")

        col_a, col_b = st.columns(2)
        col_a.metric("Tingkat Keyakinan Model", f"{hasil.confidence * 100:.1f}%")
        lawan = [v for k, v in hasil.probabilitas.items() if k != hasil.prediksi][0]
        col_b.metric("Kemungkinan Kelas Lain", f"{lawan * 100:.1f}%")

        st.progress(hasil.confidence)

        with st.expander("Lihat Detail Probabilitas"):
            for kelas, prob in hasil.probabilitas.items():
                label_k = "Pidana Khusus" if kelas == "pidana-khusus" else "Pidana Umum"
                st.write(f"{label_k}: {prob * 100:.2f}%")

        st.caption(
            "Hasil prediksi ini bersifat indikatif dan tidak menggantikan "
            "keputusan atau penilaian hukum resmi."
        )

# Keterangan di bawah
st.divider()
st.subheader("Panduan Klasifikasi Perkara")

tab1, tab2, tab3 = st.tabs(["Pidana Khusus", "Pidana Umum", "Tentang Model"])

with tab1:
    st.markdown("""
    ### Apa itu Pidana Khusus?
    Pidana khusus adalah tindak pidana yang diatur dalam undang-undang tersendiri
    di luar Kitab Undang-Undang Hukum Pidana (KUHP). Biasanya menyangkut kepentingan
    negara, masyarakat luas, atau kejahatan terorganisir.

    ### Contoh Perkara Pidana Khusus
    | Jenis | Dasar Hukum |
    |---|---|
    | Narkotika | UU No. 35 Tahun 2009 |
    | Korupsi | UU No. 31 Tahun 1999 jo. UU No. 20 Tahun 2001 |
    | Tindak Pidana Pencucian Uang | UU No. 8 Tahun 2010 |
    | ITE (Informasi & Transaksi Elektronik) | UU No. 11 Tahun 2008 jo. UU No. 19 Tahun 2016 |
    | Kekerasan Dalam Rumah Tangga (KDRT) | UU No. 23 Tahun 2004 |
    | Perlindungan Anak | UU No. 35 Tahun 2014 |
    | Terorisme | UU No. 5 Tahun 2018 |

    ### Ciri Khas
    - Ditangani oleh pengadilan dengan prosedur khusus
    - Ancaman hukuman umumnya lebih berat
    - Jaksa penuntut dari unit khusus (contoh: KPK untuk korupsi)
    """)

with tab2:
    st.markdown("""
    ### Apa itu Pidana Umum?
    Pidana umum adalah tindak pidana yang diatur dalam Kitab Undang-Undang Hukum Pidana (KUHP).
    Mencakup kejahatan sehari-hari yang terjadi di masyarakat.

    ### Contoh Perkara Pidana Umum
    | Jenis | Pasal KUHP |
    |---|---|
    | Pencurian | Pasal 362 - 367 |
    | Penganiayaan | Pasal 351 - 358 |
    | Penipuan | Pasal 378 - 395 |
    | Pembunuhan | Pasal 338 - 350 |
    | Pemerasan & Pengancaman | Pasal 368 - 371 |
    | Pemerkosaan | Pasal 285 - 288 |
    | Penggelapan | Pasal 372 - 377 |

    ### Ciri Khas
    - Ditangani oleh pengadilan negeri biasa
    - Proses hukum mengikuti KUHAP standar
    - Jaksa penuntut dari kejaksaan negeri setempat
    """)

with tab3:
    st.markdown("""
    ### Tentang Model Ini

    Model ini dibangun menggunakan algoritma **LightGBM** yang dilatih pada
    data putusan pengadilan dari 5 provinsi di Indonesia.

    ### Performa Model
    | Metrik | Nilai |
    |---|---|
    | Akurasi | 94% |
    | F1-Score | 93.6% |
    | ROC-AUC | 97.9% |
    | Data Training | 16.414 perkara |

    ### Fitur yang Digunakan
    - Jenis perkara dan kelompok umur terdakwa
    - Provinsi pengadilan
    - Pasal-pasal yang didakwakan
    - Hal yang meringankan dan memberatkan terdakwa

    ### Keterbatasan
    - Model paling akurat untuk perkara dari **Jawa Timur, Jawa Tengah,
      Jawa Barat, DKI Jakarta, dan DI Yogyakarta**
    - Untuk provinsi lain, hasil prediksi mungkin kurang akurat
    - Model bukan pengganti analisis hukum oleh profesional
    """)
