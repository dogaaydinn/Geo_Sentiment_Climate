# Design Architecture

Bu belge, projenin genel tasarım mimarisini ve her bir bileşenin işlevlerini açıklamaktadır.

## **1. Genel Yapı**

Proje, aşağıdaki katmanlara ayrılmıştır:

- **Data Layer:** Veri toplama, işleme ve depolama.
- **Logic Layer:** Veri doğrulama, iş kuralları ve analiz.
- **Presentation Layer:** Raporlama ve görselleştirme.

---

## **2. Dizayn Bileşenleri**

### **2.1 Data Layer**

- **Ham Veri (`01-data/raw`):**
    - Orijinal, işlenmemiş veri setlerini içerir.
- **İşlenmiş Veri (`01-data/processed`):**
    - Temizlenmiş ve analize hazır hale getirilmiş veri.
- **Ara Veri (`01-data/interim`):**
    - Geçici işlem aşamalarındaki veri.

---

### **2.2 Logic Layer**

- **`src/data_ingestion.py`:**
    - Ham veriyi yükler ve birleştirir.
- **`src/data_preprocessing.py`:**
    - Eksik verileri doldurur ve anormallikleri giderir.
- **`src/data_check.py`:**
    - Veri doğrulama ve profil oluşturma işlemleri.

---

### **2.3 Presentation Layer**

- **Notebooks (`02-notebooks`):**
    - EDA (Exploratory Data Analysis) ve raporlama yapılır.
- **Output (`logs`, `docs`):**
    - Hata ve işlem kayıtları ile proje belgeleri.

---

## **3. Veri İş Akışı**

1. **Veri Toplama:**
    - `src/data_ingestion.py` kullanılarak ham veri yüklenir.
2. **Veri Doğrulama:**
    - `src/data_check.py` kullanılarak veri temizliği yapılır.
3. **Ön İşleme:**
    - `src/data_preprocessing.py` kullanılarak eksik veriler doldurulur.
4. **Analiz ve Modelleme:**
    - `02-notebooks/eda_exploration.ipynb` içinde analiz gerçekleştirilir.

---

## **4. Kullanılan Araçlar**

- **Programlama Dili:** Python 3.9+
- **Kütüphaneler:**
    - `pandas`, `numpy`, `matplotlib`, `seaborn`
    - `yaml`, `logging`
- **Ortam:**
    - Jupyter Notebook

---

### **Notlar**

- Mimarinin herhangi bir kısmında eksik veya hatalı işlem olması durumunda, ilgili log dosyalarına (`logs/`) göz atın.
- Geliştirilen kodun modüler ve test edilebilir olduğundan emin olun.
