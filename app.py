
import streamlit as st
import joblib
import os
from inference_pipeline import load_artifacts, predict_klasifikasi

st.set_page_config(
    page_title="Klasifikasi Perkara Pidana",
    page_icon="scales",
    layout="centered"
)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return load_artifacts(base_dir)

artifacts = load_model()

st.title("Klasifikasi Perkara Pidana")
st.caption("Model memprediksi apakah suatu perkara termasuk pidana khusus atau pidana umum.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    jenis_perkara = st.selectbox(
        "Jenis Perkara",
        ["pid.sus", "pid.b"],
        help="pid.sus = pidana khusus, pid.b = pidana biasa/umum"
    )
    kelompok_umur = st.selectbox(
        "Kelompok Umur Terdakwa",
        ["dewasa", "dewasa-muda", "paruh-baya", "remaja", "senior", "lansia", "tidak-diketahui"]
    )
    umur = st.number_input("Umur Terdakwa (tahun)", min_value=0, max_value=100, value=25)
    provinsi = st.selectbox(
        "Provinsi Pengadilan",
        ["jatim", "jateng", "jabar", "jkt", "yogya", "lainnya"]
    )

with col2:
    pasal = st.text_area(
        "Pasal Dakwaan",
        height=100,
        placeholder="contoh: 114 112 narkotika golongan satu"
    )
    hal_meringankan = st.text_area(
        "Hal Meringankan",
        height=100,
        placeholder="contoh: sopan, menyesal, belum pernah dihukum"
    )
    hal_memberatkan = st.text_area(
        "Hal Memberatkan",
        height=100,
        placeholder="contoh: meresahkan masyarakat, tidak kooperatif"
    )

st.divider()

if st.button("Prediksi Klasifikasi", type="primary", use_container_width=True):
    if not pasal.strip():
        st.warning("Pasal dakwaan tidak boleh kosong.")
    else:
        with st.spinner("Memproses..."):
            hasil = predict_klasifikasi(
                artifacts, jenis_perkara, kelompok_umur,
                float(umur), provinsi, pasal, hal_meringankan, hal_memberatkan
            )

        st.subheader("Hasil Prediksi")

        if hasil.prediksi == "pidana-khusus":
            st.error(f"Klasifikasi: **{hasil.prediksi.upper()}**")
        else:
            st.info(f"Klasifikasi: **{hasil.prediksi.upper()}**")

        col_a, col_b = st.columns(2)
        col_a.metric("Confidence", f"{hasil.confidence * 100:.1f}%")

        lawan = [v for k, v in hasil.probabilitas.items() if k != hasil.prediksi][0]
        col_b.metric("Probabilitas Kelas Lain", f"{lawan * 100:.1f}%")

        st.progress(hasil.confidence)

        with st.expander("Detail Probabilitas"):
            for kelas, prob in hasil.probabilitas.items():
                st.write(f"{kelas}: {prob * 100:.2f}%")
