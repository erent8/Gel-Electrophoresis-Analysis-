import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_and_preprocess_image(path: str = "gel_image.png") -> np.ndarray:
    """
    Görüntüyü gri tonlamada yükler ve hafif Gaussian blur uygular.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"Görüntü yüklenemedi: '{path}'. Dosya adını ve yolunu kontrol edin."
        )

    # Hafif gürültü azaltma
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return blurred


def compute_vertical_projection(image: np.ndarray) -> np.ndarray:
    """
    Dikey piksel yoğunluğu projeksiyonunu hesaplar.
    Bantların daha net görünmesi için görüntü terslenir (invert edilir).
    """
    inverted = cv2.bitwise_not(image)
    vertical_projection = np.sum(inverted, axis=0).astype(np.float32)
    return vertical_projection


def smooth_signal(signal: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    """
    1D sinyali basit bir kutu filtre (moving average) ile yumuşatır.
    """
    kernel_size = max(5, kernel_size | 1)  # tek sayı ve en az 5
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smoothed = np.convolve(signal, kernel, mode="same")
    return smoothed


def find_lane_boundaries(
    vertical_projection: np.ndarray, min_distance: int = 20
) -> np.ndarray:
    """
    Dikey projeksiyondaki lokal minimumları (vadileri) şerit sınırı olarak tespit eder.

    Basit bir lokal minimum arama ve minimum mesafe filtresi kullanılır.
    """
    signal = vertical_projection

    # Adaptif eşik: sinyal dinamiğine göre alt limit
    s_min, s_max = float(np.min(signal)), float(np.max(signal))
    amplitude = s_max - s_min
    threshold = s_min + amplitude * 0.15  # daha derin vadileri tercih et

    candidate_indices = []
    for i in range(1, len(signal) - 1):
        if signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
            if signal[i] <= threshold:
                candidate_indices.append(i)

    # Minimum mesafe filtresi (yakın vadileri birleştir)
    filtered_minima = []
    last_idx = None
    for idx in candidate_indices:
        if last_idx is None or idx - last_idx >= min_distance:
            filtered_minima.append(idx)
            last_idx = idx
        else:
            # Yakınsa, daha derin olanı koru
            prev_idx = filtered_minima[-1]
            if signal[idx] < signal[prev_idx]:
                filtered_minima[-1] = idx
                last_idx = idx

    minima = np.array(filtered_minima, dtype=int)

    # Her durumda en sol ve en sağa sınır ekle
    if minima.size == 0:
        # Minimum bulunamazsa, kaba bir eşit bölme yap
        width = len(signal)
        approx_lane_width = max(width // 8, 40)
        minima = np.arange(approx_lane_width, width, approx_lane_width, dtype=int)

    boundaries = np.concatenate(([0], minima, [len(signal) - 1]))
    return boundaries


def draw_lane_boundaries(image: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """
    Orijinal görüntü üzerinde tespit edilen şerit sınırlarını kırmızı dikey çizgilerle çizer.
    """
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, _ = image.shape

    # İlk ve son sınırı (0 ve width-1) çizmeye gerek olmayabilir,
    # ama netlik için tüm sınırları çiziyoruz.
    for x in boundaries:
        cv2.line(color_img, (int(x), 0), (int(x), h - 1), (0, 0, 255), 1)

    # Her bir kulvarın merkezine kulvar numarasını yaz
    num_lanes = len(boundaries) - 1
    for i in range(num_lanes):
        x_start = int(boundaries[i])
        x_end = int(boundaries[i + 1])
        x_center = (x_start + x_end) // 2
        cv2.putText(
            color_img,
            str(i + 1),
            (x_center - 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return color_img


def compute_lane_intensity_profile(
    image: np.ndarray, boundaries: np.ndarray, lane_index: int = 0
) -> np.ndarray:
    """
    Verilen lane_index'e karşılık gelen şerit için yatay yoğunluk profilini hesaplar.

    - X ekseni: Satır indeksi (bant pozisyonu)
    - Y ekseni: Ortalama piksel yoğunluğu
    """
    if len(boundaries) < 2:
        raise ValueError("Şerit sınırları geçersiz veya yetersiz.")

    num_lanes = len(boundaries) - 1
    if lane_index < 0 or lane_index >= num_lanes:
        raise IndexError(f"Geçersiz şerit indeksi: {lane_index}. Toplam şerit sayısı: {num_lanes}")

    x_start = int(boundaries[lane_index])
    x_end = int(boundaries[lane_index + 1])

    if x_end <= x_start:
        raise ValueError("Seçilen şerit için geçersiz sınırlar.")

    # Şerit içindeki gri değerleri kullan
    lane_roi = image[:, x_start:x_end]

    # Her bir satır için ortalama piksel yoğunluğu
    # Jel arka planı koyu, bantlar parlak olduğu için,
    # bantlar grafikte "yukarı doğru pik" olarak çıkar.
    profile = lane_roi.mean(axis=1)
    return profile


def analyze_profile(profile: np.ndarray):
    """
    Profildeki bant piklerini (lokal maksimumlar) tespit eder ve
    kullanıcı için açıklayıcı bir metin döndürür.
    """
    if profile.size < 3:
        return np.array([], dtype=int), "Profil çok kısa, yorum yapılamıyor."

    # Profildeki gürültüyü azaltmak için yumuşatma
    kernel_size = max(5, (profile.size // 30) | 1)
    smoothed = smooth_signal(profile, kernel_size=kernel_size)

    s_min, s_max = float(smoothed.min()), float(smoothed.max())
    amplitude = s_max - s_min
    if amplitude <= 0:
        return np.array([], dtype=int), "Profil neredeyse sabit; belirgin bant görünmüyor."

    threshold = s_min + amplitude * 0.2  # yeterince güçlü pikleri al

    peak_indices = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            if smoothed[i] >= threshold:
                peak_indices.append(i)

    peaks = np.array(peak_indices, dtype=int)

    # Açıklayıcı metin
    lines = []
    lines.append("Grafik yorumu:")
    lines.append("- X ekseni: Jel üzerinde üstten alta doğru bant konumu (satır indeksi).")
    lines.append("- Y ekseni: Seçilen kulvardaki ortalama parlaklık (bant yoğunluğu).")
    lines.append(
        "- Eğrideki 'yükseliş' bant bölgesine girildiğini, 'düşüş' ise banttan çıkıldığını gösterir."
    )
    lines.append("- Tepe (pik) noktaları, jel üzerindeki parlak bantların merkezini temsil eder.")

    if len(peaks) == 0:
        lines.append(
            "\nBu kulvarda eşik üzerinde belirgin pik saptanmadı; kulvar zayıf veya boş olabilir."
        )
    else:
        lines.append(f"\nBu kulvarda yaklaşık {len(peaks)} adet belirgin bant (pik) saptandı.")
        peak_str = ", ".join(str(int(p)) for p in peaks)
        lines.append(
            f"- Bantlara karşılık gelen pik konumları (satır indeksleri): {peak_str} "
            "(üstten alta doğru)."
        )
        lines.append(
            "- Pik yüksekliği ne kadar büyükse, o banttaki molekül miktarı o kadar fazladır."
        )

    explanation = "\n".join(lines)
    return peaks, explanation


def print_menu(num_lanes: int):
    """
    Konsol ana menü çıktısını daha okunaklı bir şekilde yazdırır.
    """
    print("\n" + "=" * 60)
    title = f" Toplam {num_lanes} kulvar (şerit) bulundu "
    print(title.center(60, "="))
    print("=" * 60)
    print(" [1-9] : Kulvar seç".ljust(40) + "q : Çıkış")
    print("=" * 60 + "\n")


def print_profile_explanation(
    lane_index: int, profile: np.ndarray, peaks: np.ndarray, explanation: str
):
    """
    Seçili kulvar için profil açıklamasını çerçeveli olarak konsola yazdırır.
    Pikler varsa, her pik için satır indeksi, yoğunluk ve normalize pozisyonu
    tablo halinde özetler.
    """
    print("\n" + "-" * 60)
    print(f"Kulvar {lane_index + 1} | Pik sayısı: {len(peaks)}")
    print("-" * 60)
    print(explanation)

    # Pik özeti tablosu
    if peaks is not None and len(peaks) > 0:
        peaks = np.asarray(peaks, dtype=int)
        valid_peaks = peaks[(peaks >= 0) & (peaks < len(profile))]
        if len(valid_peaks) > 0:
            intensities = profile[valid_peaks]
            max_intensity = float(intensities.max()) if len(intensities) > 0 else 1.0
            sum_intensity = float(intensities.sum()) if len(intensities) > 0 else 1.0
            mean_intensity = float(intensities.mean())
            std_intensity = float(intensities.std()) if len(intensities) > 0 else 0.0
            length = max(1, len(profile) - 1)

            print("\nPik Özeti (jel üzerinde üstten alta doğru):")
            print("-" * 60)
            print(" Satir | Yogunluk | Max(%) | Toplam(%) | Z-skor | Pozisyon(0-1)")
            print("-" * 60)
            for row_idx, intensity in zip(valid_peaks, intensities):
                rel_to_max = (float(intensity) / max_intensity) * 100.0
                rel_to_sum = (float(intensity) / sum_intensity) * 100.0
                if std_intensity > 0:
                    z_score = (float(intensity) - mean_intensity) / std_intensity
                else:
                    z_score = 0.0
                norm_pos = float(row_idx) / float(length)
                print(
                    f" {int(row_idx):5d} | {float(intensity):9.3f} |"
                    f" {rel_to_max:7.2f} | {rel_to_sum:9.2f} | {z_score:6.2f} | {norm_pos:12.4f}"
                )
            print("-" * 60)

            # Kulvar geneli için kısa özet
            print(
                f"Toplam pik sayisi: {len(valid_peaks)}, "
                f"Toplam yogunluk: {sum_intensity:.3f}, "
                f"Ortalama: {mean_intensity:.3f}"
            )

    print("-" * 60 + "\n")


def save_lane_report(
    lane_index: int, profile: np.ndarray, peaks: np.ndarray, report_path: str
) -> None:
    """
    Seçili kulvar için pik bilgilerini CSV formatında rapor dosyasına kaydeder.

    CSV kolonları:
    lane_index, peak_row, peak_intensity, rel_intensity_max_percent,
    rel_intensity_sum_percent, z_score, normalized_position
    """
    # Raporlanacak pik yoksa yine de kulvarın boş olduğunu işaretle
    if peaks is None or len(peaks) == 0:
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"{lane_index + 1},,,,\n")
        return

    peaks = np.asarray(peaks, dtype=int)
    valid_peaks = peaks[(peaks >= 0) & (peaks < len(profile))]
    if len(valid_peaks) == 0:
        # Tüm pikler geçersiz ise, boş satır yaz
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"{lane_index + 1},,,,\n")
        return

    intensities = profile[valid_peaks]
    max_intensity = float(intensities.max()) if len(intensities) > 0 else 1.0
    sum_intensity = float(intensities.sum()) if len(intensities) > 0 else 1.0
    mean_intensity = float(intensities.mean())
    std_intensity = float(intensities.std()) if len(intensities) > 0 else 0.0
    length = max(1, len(profile) - 1)

    with open(report_path, "a", encoding="utf-8") as f:
        for row_idx, intensity in zip(valid_peaks, intensities):
            rel_to_max = (float(intensity) / max_intensity) * 100.0
            rel_to_sum = (float(intensity) / sum_intensity) * 100.0
            if std_intensity > 0:
                z_score = (float(intensity) - mean_intensity) / std_intensity
            else:
                z_score = 0.0
            norm_pos = float(row_idx) / float(length)
            f.write(
                f"{lane_index + 1},{int(row_idx)},{float(intensity):.3f},"
                f"{rel_to_max:.2f},{rel_to_sum:.2f},{z_score:.3f},{norm_pos:.4f}\n"
            )


def update_lane_view(
    lane_index: int,
    gray_blurred: np.ndarray,
    boundaries: np.ndarray,
    num_lanes: int,
    report_path: str,
    output_dir: str,
) -> None:
    """
    Seçilen kulvarın profilini hesaplar, grafiği günceller, konsola açıklama ve tablo
    yazar, ayrıca CSV raporunu ve profil görüntüsünü kaydeder.
    """
    profile = compute_lane_intensity_profile(
        gray_blurred, boundaries, lane_index=lane_index
    )
    peaks, explanation = analyze_profile(profile)
    profile_img = plot_profile_to_image(profile, peaks=peaks)

    # Güncel kulvar numarasını ve pik sayısını görüntü üzerine yaz
    cv2.putText(
        profile_img,
        f"Kulvar: {lane_index + 1} | Pik sayisi: {len(peaks)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    print(f"Kulvar {lane_index + 1} profili gosteriliyor...")
    print_profile_explanation(lane_index, profile, peaks, explanation)

    # Raporu ve profil görüntüsünü kaydet
    save_lane_report(lane_index, profile, peaks, report_path)
    print(f"Rapor güncellendi: {report_path}")

    profile_path = os.path.join(output_dir, f"profile_lane_{lane_index + 1}.png")
    cv2.imwrite(profile_path, profile_img)
    print(f"Profil görüntüsü kaydedildi: {profile_path}")

    # Profil penceresini güncelle
    cv2.imshow("Serit Yogunluk Profili", profile_img)


def plot_profile_to_image(profile: np.ndarray, peaks=None) -> np.ndarray:
    """
    Matplotlib ile yoğunluk profilini çizer ve sonucu bir numpy görüntüsüne dönüştürür.
    Böylece profil, cv2.imshow ile gösterilebilir.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(profile, label="Yoğunluk Profili")

    # Pikleri grafikte kırmızı işaretçilerle göster
    if peaks is not None and len(peaks) > 0:
        peaks = np.asarray(peaks, dtype=int)
        valid_peaks = peaks[(peaks >= 0) & (peaks < len(profile))]
        if len(valid_peaks) > 0:
            ax.scatter(
                valid_peaks,
                profile[valid_peaks],
                color="red",
                s=30,
                label="Bant Pikleri",
                zorder=3,
            )

    # Not: Bazı Windows kurulumlarında Türkçe karakterler (Ş, Ğ, İ, ı, ç, ö, ü)
    # pencere başlıklarında bozulabildiği için, grafik başlık ve eksenleri
    # ASCII karakterlerle yazıyoruz.
    ax.set_title("Serit Yogunluk Profili")
    ax.set_xlabel("Satir Indeksi (Bant pozisyonu)")
    ax.set_ylabel("Ortalama Piksel Yogunlugu")

    if peaks is not None and len(peaks) > 0:
        ax.legend(loc="best")
    fig.tight_layout()

    # Figure'ı RGBA numpy array'e dönüştür (backend-bağımsız yöntem)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plot_img = plot_img.reshape((h, w, 4))

    # RGBA -> BGR (cv2.imshow ile uyumlu)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)

    plt.close(fig)
    return plot_img


def main():
    # 1) Görüntü yükleme ve ön işleme
    # Script ve görüntü aynı klasördeyse sadece dosya adını kullanmak yeterli:
    # lane_analysis.py  ->  gel_image.png  (aynı dizin)
    try:
        gray_blurred = load_and_preprocess_image("gel_image.png")
    except FileNotFoundError as e:
        print(e)
        print("Lütfen 'gel_image.png' dosyasının script ile aynı dizinde olduğundan emin olun.")
        return

    # 2) Dikey projeksiyon ve şerit sınırlarının tespiti
    vertical_projection = compute_vertical_projection(gray_blurred)

    width = gray_blurred.shape[1]
    kernel_size = max(15, (width // 50) | 1)
    smoothed_projection = smooth_signal(vertical_projection, kernel_size=kernel_size)

    min_distance = max(20, width // 20)
    boundaries = find_lane_boundaries(smoothed_projection, min_distance=min_distance)

    # Şerit sınırlarını görüntü üzerinde çiz
    lane_image = draw_lane_boundaries(gray_blurred, boundaries)

    # --- GÜNCELLEME BURADA BAŞLIYOR ---

    # Tespit edilen toplam kulvar sayısı
    num_lanes = len(boundaries) - 1
    current_lane_index = 0

    # Rapor dosyasını ve klasörünü hazırla
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "gel_report.csv")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            "lane_index,peak_row,peak_intensity,"
            "rel_intensity_max_percent,rel_intensity_sum_percent,z_score,normalized_position\n"
        )

    # Şerit sınırları çizilmiş jel görüntüsünü de kaydet
    lane_img_path = os.path.join(output_dir, "lane_boundaries.png")
    cv2.imwrite(lane_img_path, lane_image)
    print(f"Serit sinirlari görüntüsü kaydedildi: {lane_img_path}")

    if num_lanes <= 0:
        print("Hata: Görüntüde hiçbir kulvar (şerit) tespit edilemedi.")
        return

    print_menu(num_lanes)


    # 3) İlk şeritin yoğunluk profilini hesapla, göster ve kaydet
    update_lane_view(
        current_lane_index,
        gray_blurred,
        boundaries,
        num_lanes,
        report_path,
        output_dir,
    )

    # 4) Jel görüntüsü penceresini göster
    cv2.imshow("Jel Goruntusu - Serit Sinirlari", lane_image)

    # İnteraktif döngü: Tuş basılmasını bekler (waitKey(0))
    while True:
        # 0: Bir tuşa basılana kadar süresiz bekle
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            print("Program sonlandırıldı.")
            break
        
        # Sayı tuşlarını kontrol et (1-9 arası)
        # chr(key) -> basılan tuşun karakter karşılığı ('1', '2' vb.)
        if ord("1") <= key <= ord("9"):
            # '1' tuşu -> index 0
            # '2' tuşu -> index 1
            lane_idx = int(chr(key)) - 1
            
            if lane_idx < num_lanes:
                current_lane_index = lane_idx
                update_lane_view(
                    current_lane_index,
                    gray_blurred,
                    boundaries,
                    num_lanes,
                    report_path,
                    output_dir,
                )
            else:
                print(f"Gecersiz kulvar! Sadece {num_lanes} kulvar var (1-{num_lanes} arasi secim yapin).")

        # Sağ/sol yön tuşları veya A/D ile kulvarlar arasında gezin
        elif key in (ord("d"), ord("D"), 83):  # sağ yön veya D
            current_lane_index = (current_lane_index + 1) % num_lanes
            update_lane_view(
                current_lane_index,
                gray_blurred,
                boundaries,
                num_lanes,
                report_path,
                output_dir,
            )
        elif key in (ord("a"), ord("A"), 81):  # sol yön veya A
            current_lane_index = (current_lane_index - 1) % num_lanes
            update_lane_view(
                current_lane_index,
                gray_blurred,
                boundaries,
                num_lanes,
                report_path,
                output_dir,
            )
        
        # Diğer tuşlara basılırsa (örn: Shift, Ctrl) döngüye devam et
        else:
            continue
            
    # --- GÜNCELLEME BURADA BİTİYOR ---

    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
