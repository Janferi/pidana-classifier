
import joblib
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Literal

# Konstanta
PROVINSI_ORDER      = ["jabar", "jateng", "jatim", "jkt", "yogya"]
PROVINSI_VALID      = set(PROVINSI_ORDER)
KELOMPOK_UMUR_VALID = {"remaja", "dewasa-muda", "dewasa", "paruh-baya",
                        "senior", "lansia", "tidak-diketahui"}
JENIS_PERKARA_VALID = {"pid.sus", "pid.b"}


@dataclass
class HasilPrediksi:
    prediksi    : str
    confidence  : float
    probabilitas: dict


def _validasi_input(jenis_perkara: str, kelompok_umur: str, umur: float, provinsi: str) -> None:
    if jenis_perkara not in JENIS_PERKARA_VALID:
        raise ValueError(
            f"jenis_perkara harus salah satu dari {JENIS_PERKARA_VALID}, "
            f"dapat: '{jenis_perkara}'"
        )
    if kelompok_umur not in KELOMPOK_UMUR_VALID:
        raise ValueError(
            f"kelompok_umur harus salah satu dari {KELOMPOK_UMUR_VALID}, "
            f"dapat: '{kelompok_umur}'"
        )
    if not isinstance(umur, (int, float)) or umur < 0:
        raise ValueError(f"umur harus angka non-negatif, dapat: {umur}")


def _bangun_fitur_struktural(artifacts: dict, jenis_perkara: str,
                              kelompok_umur: str, umur: float,
                              provinsi: str) -> sp.csr_matrix:
    enc_jp   = artifacts["le_jenis"].transform([jenis_perkara])[0]
    enc_umur = artifacts["oe_umur"].transform([[kelompok_umur]])[0][0]
    umur_sc  = artifacts["scaler_umur"].transform([[umur]])[0][0]
    is_rem   = int(kelompok_umur == "remaja")
    prov_enc = [int(provinsi == p) for p in PROVINSI_ORDER]

    return sp.csr_matrix([[enc_jp, enc_umur, umur_sc, is_rem] + prov_enc])


def _bangun_fitur_teks(artifacts: dict, pasal: str,
                        hal_meringankan: str,
                        hal_memberatkan: str) -> tuple:
    return (
        artifacts["tfidf_pasal"].transform([pasal]),
        artifacts["tfidf_ringan"].transform([hal_meringankan]),
        artifacts["tfidf_berat"].transform([hal_memberatkan]),
    )


def load_artifacts(model_dir: str) -> dict:
    """
    Muat semua artifacts model dari direktori.

    Parameters
    ----------
    model_dir : str
        Path ke folder berisi file .pkl

    Returns
    -------
    dict
        Semua artifacts yang dibutuhkan untuk inferensi
    """
    import os

    keys = [
        "lgbm_tuned_final",
        "label_encoder_target",
        "le_jenis",
        "oe_umur",
        "scaler_umur",
        "tfidf_pasal",
        "tfidf_ringan",
        "tfidf_berat",
    ]

    return {
        key: joblib.load(os.path.join(model_dir, f"{key}.pkl"))
        for key in keys
    }


def predict_klasifikasi(
    artifacts       : dict,
    jenis_perkara   : Literal["pid.sus", "pid.b"],
    kelompok_umur   : str,
    umur            : float,
    provinsi        : str,
    pasal           : str,
    hal_meringankan : str,
    hal_memberatkan : str,
) -> HasilPrediksi:
    """
    Prediksi klasifikasi perkara pidana.

    Parameters
    ----------
    artifacts       : output dari load_artifacts()
    jenis_perkara   : "pid.sus" atau "pid.b"
    kelompok_umur   : salah satu dari KELOMPOK_UMUR_VALID
    umur            : usia terdakwa (float >= 0), 0 jika tidak diketahui
    provinsi        : salah satu dari PROVINSI_VALID, atau string lain
    pasal           : teks isi atau nomor pasal dakwaan
    hal_meringankan : teks hal yang meringankan terdakwa
    hal_memberatkan : teks hal yang memberatkan terdakwa

    Returns
    -------
    HasilPrediksi
        prediksi, confidence, dan probabilitas per kelas
    """
    _validasi_input(jenis_perkara, kelompok_umur, umur, provinsi)

    mat_str                = _bangun_fitur_struktural(
                                artifacts, jenis_perkara, kelompok_umur, umur, provinsi
                             )
    mat_ps, mat_ri, mat_be = _bangun_fitur_teks(
                                artifacts, pasal, hal_meringankan, hal_memberatkan
                             )
    X_input = sp.hstack([mat_str, mat_ps, mat_ri, mat_be])

    model      = artifacts["lgbm_tuned_final"]
    le_target  = artifacts["label_encoder_target"]
    pred_label = model.predict(X_input)[0]
    pred_proba = model.predict_proba(X_input)[0]
    label      = le_target.inverse_transform([pred_label])[0]

    return HasilPrediksi(
        prediksi     = label,
        confidence   = round(float(pred_proba[pred_label]), 4),
        probabilitas = {
            le_target.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(pred_proba)
        },
    )
