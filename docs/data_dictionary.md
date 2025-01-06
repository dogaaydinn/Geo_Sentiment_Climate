# Data Dictionary

Bu belge, projede kullanılan veri setindeki sütunların anlamlarını ve veri türlerini açıklamaktadır.

## Genel Bilgiler

- **Kaynak Veriler:** EPA Hava Kalitesi Verileri
- **Sütun Sayısı:** 15
- **Veri Türleri:** Metin, Sayısal, Tarih

### **1. Ortak Sütunlar**

| Sütun Adı     | Açıklama                     | Veri Türü | Örnek Değer  |
|---------------|------------------------------|-----------|--------------|
| `date`        | Ölçüm tarihi                 | DateTime  | 2023-07-15   |
| `state_fips`  | ABD eyalet kodu              | Integer   | 04 (Arizona) |
| `county_fips` | ABD ilçe kodu                | Integer   | 013          |
| `aqi`         | Günlük Hava Kalitesi Endeksi | Integer   | 45           |
| `units`       | Ölçüm birimi                 | String    | µg/m³        |

### **2. Parametreye Özel Sütunlar**

- Her bir hava kirletici parametresi için spesifik sütunlar.

| Sütun Adı                              | Açıklama                                 | Veri Türü | Örnek Değer |
|----------------------------------------|------------------------------------------|-----------|-------------|
| `Daily Max 1-hour SO2 Concentration`   | Günlük maksimum 1 saatlik SO2 yoğunluğu  | Float     | 0.003       |
| `Daily Max 8-hour Ozone Concentration` | Günlük maksimum 8 saatlik ozon yoğunluğu | Float     | 0.042       |
| `Daily Max 8-hour CO Concentration`    | Günlük maksimum 8 saatlik CO yoğunluğu   | Float     | 0.9         |
| `Daily Mean PM2.5 Concentration`       | Günlük ortalama PM2.5 yoğunluğu          | Float     | 15.6        |

### **3. Eksik ve Null Veriler**

- Bazı sütunlarda eksik veya null değerler bulunabilir.
- Eksik veri için doldurma stratejileri:
    - `mean` veya `median` ile doldurma.
    - Spesifik tarih aralıklarında interpolasyon.

---

### **Notlar**

- Daha fazla sütun veya özel dönüşüm ihtiyacı olması durumunda `src/data_preprocessing.py` içinde ilgili sütunlar ele
  alınabilir.
