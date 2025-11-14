## Jel Görüntüsü Şerit Analizi (Türkçe)

Bu proje, bir jel görüntüsündeki (örneğin DNA/protein jel elektroforezi) **şeritleri (kulvarları)** ve bu şeritler içindeki **bantları** otomatik olarak analiz etmek için hazırlanmış bir Python aracıdır.  
`lane_analysis.py` betiği:

- **Gri tonlamalı jel görüntüsünü** yükler ve yumuşatır.
- Dikey projeksiyon ile **şerit sınırlarını** otomatik tespit eder.
- Her şerit için **yoğunluk profili** çıkarır ve bantlara karşılık gelen **pik noktalarını** bulur.
- Sonuçları hem **görsel pencerelerde** hem de **CSV raporu** ve **PNG çıktıları** olarak kaydeder.

---

### Gereksinimler

- Python 3.8+ (3.10 civarı önerilir)
- Aşağıdaki paketler:
  - `opencv-python`
  - `numpy`
  - `matplotlib`

Kurulum (örnek):

```bash
pip install opencv-python numpy matplotlib
```

---

### Proje Yapısı

- `lane_analysis.py` → Ana analiz betiği
- `gel_image.png` → Analiz edilecek jel görüntüsü (aynı klasörde olmalı)
- `outputs/` → Çıktı klasörü (betik çalışınca otomatik oluşur)
  - `lane_boundaries.png` → Şerit sınırları çizilmiş jel görüntüsü
  - `profile_lane_X.png` → Her kulvar için yoğunluk profili grafiği (X = kulvar numarası)
  - `gel_report.csv` → Tüm kulvar ve bantlar için sayısal özet rapor

---

### Çalıştırma

1. `lane_analysis.py` ve `gel_image.png` aynı klasörde olsun.
2. Gerekli Python paketlerini kurduğundan emin ol.
3. Komut satırında proje klasörüne gel ve:

```bash
python lane_analysis.py
```

---

### Etkileşimli Kullanım

Betik çalıştığında iki pencere açılır:

- `Jel Goruntusu - Serit Sinirlari`  
  Şerit sınırları kırmızı çizgilerle gösterilen jel görüntüsü.

- `Serit Yogunluk Profili`  
  Seçili kulvar için bant yoğunluk profilini gösteren grafik.

Klavye kontrolleri:

- `1`–`9` → İlgili kulvarın profilini göster (varsa).
- `D` / `d` veya **sağ yön tuşu** → Bir sonraki kulvara geç.
- `A` / `a` veya **sol yön tuşu** → Bir önceki kulvara geç.
- `q` → Programdan çık.

Her kulvar seçildiğinde:

- Konsolda, grafik yorumu ve **pik özeti tablosu** gösterilir.
- `outputs/profile_lane_X.png` adıyla profil grafiği kaydedilir.
- `outputs/gel_report.csv` dosyasına ilgili satırlar eklenir/güncellenir.

---

### `gel_report.csv` İçeriği

CSV dosyasında her satır bir **kulvardaki tek bir bant (pik)** için bilgileri tutar:

- `lane_index` → Kulvar numarası (1’den başlar)
- `peak_row` → Bantın jel üzerindeki satır indeksi (üstten alta)
- `peak_intensity` → Pik yoğunluk değeri
- `rel_intensity_max_percent` → Aynı kulvardaki en güçlü banda göre yüzdesi
- `rel_intensity_sum_percent` → Kulvardaki tüm bantların toplamına göre yüzdesi
- `z_score` → Pik yoğunluğunun, o kulvardaki diğer pike göre Z-skoru
- `normalized_position` → Jel yüksekliği boyunca 0–1 arası normalize edilmiş konum

Bu dosyayı Excel, R, Python vb. ile kolayca açıp ileri analiz yapabilirsin.

---

### Notlar ve Öneriler

- Farklı jeller için **eşik değerleri ve kernel boyutları** kod içinde ayarlanabilir.
- Eğer şeritler düzgün ayrılmıyorsa, görüntüyü önceden kırpmak veya kontrast ayarı yapmak yardımcı olabilir.
- İleride istenirse, farklı görüntüleri toplu işlemek için basit bir döngü eklenebilir.


