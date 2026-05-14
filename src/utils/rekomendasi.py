def generate_rekomendasi(kategori_makanan: list, dds_score: int, status_anemia: str,
                         status_tb_u: str, kelas_ekonomi: str, status_sanitasi: str,
                         status_lahir: str) -> str:
    saran = []

    # 🥗 Rekomendasi berdasarkan DDS Score
    if dds_score < 4:
        if kelas_ekonomi in ["Kelas Atas", "Kelas Menengah"]:
            saran.append("Coba tambahkan makanan seperti telur, ikan, dan susu ke dalam menu harian.")
        else:
            saran.append("Coba tambahkan protein murah seperti tempe, tahu, dan sayur-sayuran hijau lokal.")
    else:
        saran.append("Menu makanan sudah cukup beragam, pertahankan variasi bahan makanan setiap hari.")

    # 📏 Rekomendasi berdasarkan status TB/U
    if status_tb_u in ["Sangat Pendek", "Pendek"]:
        if kelas_ekonomi == "Kelas Atas":
            saran.append("Konsultasikan ke dokter dan pertimbangkan makanan tinggi protein dan kalsium seperti susu dan daging.")
        else:
            saran.append("Perbanyak konsumsi protein nabati, sayur, dan buah lokal yang mudah didapat.")

    # 🩸 Rekomendasi berdasarkan anemia
    if status_anemia == "Ya":
        if kelas_ekonomi in ["Kelas Atas", "Kelas Menengah"]:
            saran.append("Tambahkan makanan tinggi zat besi seperti hati ayam, daging merah, dan bayam.")
        else:
            saran.append("Perbanyak konsumsi daun singkong, bayam, dan kacang-kacangan.")
    else:
        saran.append("Status anemia baik, pertahankan pola makan seimbang kaya zat besi.")

    # 🚿 Rekomendasi berdasarkan sanitasi
    if status_sanitasi == "Buruk":
        saran.append("Jaga kebersihan dengan lebih rutin seperti mencuci tangan, menyikat gigi, dan BAB di toilet sehat.")
    elif status_sanitasi == "Cukup":
        saran.append("Kebersihan sudah cukup baik, tapi bisa lebih ditingkatkan terutama sebelum makan dan setelah BAB.")
    elif status_sanitasi == "Baik":
        saran.append("Sanitasi kamu bagus! Pertahankan dan tingkatkan ya!")

    # 👶 Rekomendasi berdasarkan kondisi lahir
    saran.append(f"Kondisi lahir anak: {status_lahir}")

    return "\n".join(saran)

