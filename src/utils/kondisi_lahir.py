def generate_rekomendasi_lahir(berat_lahir_gram, tinggi_lahir_cm, status_tb_u=None, status_anemia=None, kelas_ekonomi=None):
    """
    Menentukan kondisi lahir anak berdasarkan berat dan tinggi badan saat lahir.
    Standar WHO (disesuaikan Kemenkes).
    """
    if berat_lahir_gram < 2500 and tinggi_lahir_cm < 48:
        kondisi = "Berat & Tinggi Lahir Kurang"
        kategori = "kurang"
    elif berat_lahir_gram < 2500:
        kondisi = "Berat Lahir Kurang"
        kategori = "kurang"
    elif tinggi_lahir_cm < 48:
        kondisi = "Tinggi Lahir Kurang"
        kategori = "kurang"
    elif berat_lahir_gram > 4000:
        kondisi = "Berat Lahir Lebih"
        kategori = "lebih"
    else:
        kondisi = "Normal"
        kategori = "normal"

    rekomendasi = []
    if kategori == "kurang":
        rekomendasi.append(
            "Perhatikan asupan energi dan protein sejak dini, terutama dari telur, ikan, dan daging ayam."
        )
        if kelas_ekonomi == "Kelas Bawah":
            rekomendasi.append("Manfaatkan sumber protein lokal seperti tempe, tahu, dan ikan kecil.")
        if status_anemia == "Ya":
            rekomendasi.append("Tambahkan makanan kaya zat besi seperti bayam dan hati ayam.")
    elif kategori == "lebih":
        rekomendasi.append(
            "Batasi konsumsi gula dan lemak berlebih, serta tingkatkan aktivitas fisik rutin."
        )
    else:
        rekomendasi.append(
            "Pertahankan pola makan seimbang dan lakukan pemantauan pertumbuhan secara rutin."
        )

    return {
        "kondisi": kondisi,
        "kategori": kategori,
        "rekomendasi": " ".join(rekomendasi)
    }


