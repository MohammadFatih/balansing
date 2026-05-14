from ultralytics import YOLO
from src.utils.dds import calculate_dds
from src.utils.anemia import deteksi_anemia
from src.zscore.hitung_tb_u import hitung_z_score_tb_u
from src.utils.ekonomi import klasifikasi_ekonomi
from src.utils.rekomendasi import generate_rekomendasi
from src.utils.sanitasi import deteksi_sanitasi
from src.utils.kondisi_lahir import generate_rekomendasi_lahir

# === 0. INPUT USER ===
user_input = {
    "id": "U001",
    "tinggi_cm": 79,
    "usia_bulan": 24,
    "jenis_kelamin": "L",
    "berat_lahir" : 2400,
    "tinggi_lahir" : 47,
    "image_path": r"C:\Users\Mohammad Fatih\Documents\Fatih\Research\balansing-ai_baru\data\sample\foto_makanan.png",
    "status_anemia": {
        "lemas": True,
        "riwayat": False,
        "konjungtiva_pucat": True,
        "kuku_pucat": False
    },
    "pengeluaran_perkapita": 1800000,
    "sanitasi": {
        "sikat_gigi_harian": True,
        "waktu_sikat_gigi": {
            "sebelum_makan_pagi": False,
            "setelah_makan_pagi": True,
            "sesudah_makan_siang": False,
            "saat_mandi_pagi": True,
            "saat_mandi_sore": True,
            "sebelum_tidur": True
    },
        "cuci_tangan_harian": True,
        "waktu_cuci_tangan": {
            "sebelum_makan": True,
            "setelah_makan": True,
            "sebelum_bak_bab": False,
            "setelah_bak_bab": True
    },
        "bab_di_toilet": True,
        "air_mineral_untuk_minum_masak": False
    }
}

# === 1. PREDIKSI KATEGORI MAKANAN DENGAN YOLO ===
model = YOLO("C:/Users/Mohammad Fatih/Documents/Fatih/Research/balansing-ai_baru/data/train30/runs/detect/test_fixed2/weights/901.pt")
results = model.predict(source=user_input["image_path"], conf=0.2,iou=0.45, save=True)

# Mapping label YOLO ke 7 kategori DDS
label_to_dds = {
    "daging": "daging",
    "makanan_berpati": "makanan_berpati",
    "telur": "telur",
    "buah_sayur_lainnya": "buah_sayur_lainnya",
    "kacang_legume": "kacang_legume",
    "buah_sayur_vitA": "buah_sayur_vitA",
    "produk_susu": "produk_susu" 
}

# Ambil label dari YOLO dan mapping ke DDS
detected_raw = []
for r in results:
    detected_raw = [model.names[int(cls_id)] for cls_id in r.boxes.cls]

detected_labels = [label_to_dds.get(label, label) for label in detected_raw]
detected_labels = [label.strip() for label in detected_labels]
print("Kategori makanan terdeteksi (dari model):", detected_labels)

# === 2. GABUNGKAN DENGAN INPUT MANUAL (JIKA ADA) ===
valid_dds = {
    "makanan_berpati",
    "daging",
    "telur",
    "produk_susu",
    "kacang_legume",
    "buah_sayur_vitA",
    "buah_sayur_lainnya"
}

# Tambahkan input_manual dari user_input jika ada
manual_input = [item.strip().lower() for item in user_input.get("input_manual", []) if item.strip().lower() in valid_dds]
detected_labels.extend(manual_input)

# Cek sisa kategori DDS yang belum terdeteksi
kategori_tidak_terdeteksi = [k for k in valid_dds if k not in detected_labels]

# Jika masih ada yang belum terdeteksi, minta input dari terminal
if kategori_tidak_terdeteksi:
    print(f"Ada {len(kategori_tidak_terdeteksi)} kategori DDS belum terdeteksi: {kategori_tidak_terdeteksi}")
    input_str = input("Tambahkan kategori (jika ada, pisahkan dengan koma atau tekan Enter jika tidak ada):\n> ").strip()

    if input_str:  # hanya proses jika user benar-benar mengisi
        tambahan_terminal = [x.strip().lower() for x in input_str.split(",") if x.strip().lower() in valid_dds]
        detected_labels.extend(tambahan_terminal)
    if not input_str:
        print("Tidak ada tambahan manual. Menggunakan hasil deteksi YOLO sepenuhnya.")


# Hapus duplikat
detected_labels = list(set(detected_labels))
print("Kategori makanan akhir (deteksi + input manual):", detected_labels)

# === 3. HITUNG DDS SCORE ===
dds_score = calculate_dds(detected_labels)
print(f"DDS Score: {dds_score} dari {detected_labels}")

# === 4. HITUNG Z-SCORE TB/U ===
z_score, status_tb_u = hitung_z_score_tb_u(
    usia_bulan=user_input["usia_bulan"],
    tinggi_cm=user_input["tinggi_cm"],
    jenis_kelamin=user_input["jenis_kelamin"]
)
print(f"Z-score TB/U: {z_score:.2f} → {status_tb_u}")

# === 5. DETEKSI STATUS ANEMIA ===
status_anemia = deteksi_anemia(
    user_input["status_anemia"]["lemas"],
    user_input["status_anemia"]["riwayat"],
    user_input["status_anemia"]["konjungtiva_pucat"],
    user_input["status_anemia"]["kuku_pucat"]
)
print("Status Anemia:", status_anemia)

#===6. Deteksi Sanitasi ===
status_sanitasi = deteksi_sanitasi(user_input["sanitasi"])
print(f"Status Sanitasi: {status_sanitasi}")

# === 7. KLASIFIKASI EKONOMI ===
kelas_ekonomi = klasifikasi_ekonomi(user_input["pengeluaran_perkapita"])
print(f"Kelas Ekonomi   : {kelas_ekonomi}")

# === 8. DETEKSI KONDISI LAHIR (BARU) ===
status_lahir = generate_rekomendasi_lahir(
    berat_lahir_gram=user_input["berat_lahir"],
    tinggi_lahir_cm=user_input["tinggi_lahir"]
)
print(f"Kondisi Lahir: {status_lahir}")

# === 9. GENERATE REKOMENDASI ===
rekomendasi = generate_rekomendasi(
    kategori_makanan=detected_labels,
    dds_score=dds_score,
    status_anemia=status_anemia,
    status_tb_u=status_tb_u,
    kelas_ekonomi=kelas_ekonomi,
    status_sanitasi=status_sanitasi,
    status_lahir=status_lahir
)

# === 8. OUTPUT FINAL ===
print("\n=== HASIL SISTEM BALANSING-AI ===")
print(f"ID User        : {user_input['id']}")
print(f"Z-score TB/U   : {z_score:.2f} → {status_tb_u}")
print(f"DDS Score      : {dds_score} (dari {detected_labels})")
print(f"Status Anemia  : {status_anemia}")
print(f"Kelas Ekonomi  : {kelas_ekonomi}")
print(f"Status Sanitasi : {status_sanitasi}")
print(f"kondisi lahir : {status_lahir}")

print("\n=== REKOMENDASI  ===")
print(rekomendasi)
