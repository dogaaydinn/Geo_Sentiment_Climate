# <Geo_Sentiment_Climate> (Advanced Data Analytics Project)

## 1. Overview

Welcome to the **<Geo_Sentiment_Climate>** repository. This project aims to collect, clean, and analyze air quality data (CO, SO2, NO2, O3, PM2.5, etc.) from the EPA (Environmental Protection Agency) and other sources. We'll unify the datasets, handle missing values/outliers, and eventually perform advanced analytics/ML on them.

### Key Objectives:

1. **Data Ingestion & Preprocessing**: 
   - Gather raw CSV files from multiple states/parameters.
   - Standardize column names, unify param (co, so2, no2, pm25, o3).
2. **Exploratory Data Analysis**:
   - Visualize time series, geospatial distributions, etc.
3. **Feature Engineering & Modeling** (optional):
   - If time permits, build a forecast or regression model.

## 2. Repository Structure

```text
Geo_Sentiment_Climate/
  ├─ 00-config/
my_advanced_project/
  ├─ 00-config/
  │   ├─ settings.yml
  │   ├─ credentials_template.yml
  │   └─ ...
  ├─ 01-data/
  │   ├─ raw/
  │   │   └─ # Ham veri CSV veya ZIP dosyaları (EPA CO, NO2, vb.)
  │   ├─ interim/
  │   │   └─ # Ara işlenmiş veriler, geçici dosyalar
  │   └─ processed/
  │       └─ # Temizlenmiş / birleştirilmiş veriler (epa_long.csv, epa_wide.csv vb.)
  ├─ 02-notebooks/
  │   ├─ 01_data_check.ipynb
  │   ├─ 02_data_ingestion.ipynb
  │   ├─ 03_eda_exploration.ipynb
  │   ├─ 04_feature_engineering.ipynb
  │   ├─ 05_modeling.ipynb
  │   ├─ 06_evaluation_and_viz.ipynb
  │   └─ ...
  ├─ 03-src/
  │   ├─ data_check.py
  │   ├─ data_ingestion.py
  │   ├─ data_preprocessing.py
  │   ├─ utils/
  │   │   └─ config_loader.py
  │   └─ __init__.py
  ├─ 04-logs/
  │   └─ eda_exploration.log
  ├─ 05-docs/
  │   ├─ references/
  │   │   └─ # Makale PDF'leri, link açıklamaları, vb.
  │   ├─ design_architecture.md
  │   └─ data_dictionary.md
  ├─ .gitignore
  ├─ README.md
  ├─ requirements.txt (veya environment.yml)
  └─ LICENSE (opsiyonel)

```

### Directory Summary

-   **00-config/**: Project configuration (settings, credentials template).
- Amaç: Proje ayar dosyalarını (YAML, JSON, vs.) burada tutmak.
settings.yml: Örneğin “veri kaynak yolu, parametre listesi, vs.” gibi proje konfigürasyonunu tutabilirsin.
credentials_template.yml: Gizli anahtar/token kullanacaksan, “template” dosyayı buraya koyarsın, gerçek kimlik bilgilerini .gitignore’a alırsın.
Bu sayede projede hard-coded yollar veya parametreler yerine “config dosyası” yaklaşımı uygulayarak profesyonel bir izlenim bırakırsınız.
01-data/raw/
Tamamen ham veri (EPA CSV/ZIP’ler) alt klasörlerde durur.
Örnek alt klasörler: epa-co-2023/, epa-so2-2022/, vb.
.gitignore: Büyük dosyaları Git’e göndermemeniz önerilir.
01-data/interim/
Ara işlenmiş (ör. ilk rename, mini temizlik) verileri burada tutabilirsin.
Bazı projelerde temp/ veya intermediate/ ismi veriliyor.
01-data/processed/
Nihai veya analiz-ready dosyalar buraya çıkıyor: epa_long.csv, epa_wide.csv, vb.
Model için finalize edilmiş data veya “cleaned.csv” de burada olabilir.
Bu şekilde rahat anlaşılıyor: “raw → interim → processed”.

-   **01-data/**:
    -   `raw/`: Original CSV files from EPA, etc.
    -   `interim/`: Temporary intermediate files.
    -   `processed/`: Cleaned, standardized final CSVs (e.g. epa_long.csv).
-   **02-notebooks/**: Jupyter notebooks for data check, ingestion, EDA, modeling.
- Notebooks’ları numaralandırıp isim vererek kronolojik bir yapı sağlarsınız:

01_data_check.ipynb: CSV’lerin kontrolü, shape/columns, mini testler.
02_data_ingestion.ipynb: Tüm ingestion/konsolidasyon rename mantığı (ya da src/data_ingestion.py’ye çağırabilirsiniz).
03_eda_exploration.ipynb: Görsel grafikler, korelasyon, anomali tespiti.
04_feature_engineering.ipynb: Değişken üretme, unit conversion, demografi merge.
05_modeling.ipynb: Zaman serisi/ML modeli, param tuning.
06_evaluation_and_viz.ipynb: Sonuç görselleştirme, tablo, metrikler.
Böylece Notebook’lar adım adım gider, “Hangi dosya ne yapıyordu?” sorusu minimize olur. Siz ilgili Notebook’larda kendi kod/hücrelerinizi geliştirirsiniz.
-   **03-src/**: Python scripts for ingestion, preprocessing, utilities, etc.
- Burada Python script dosyalarınız durur. Daha modüler yaklaşırsanız, Notebook’lar sadece “script fonksiyonlarını çağıran” alanlar olur:

data_check.py: “Glob .csv” + shape kontrol + raporlama.
data_ingestion.py: rename_map, param dedect, concat + pivot.
data_preprocessing.py: null doldurma, outlier fix, units conversion vs.
utils/: yard. fonksiyonlar (common.py, logger.py, vs.)
__init__.py: (Python package geleneği, opsiyonel ama profesyonel durur).
Şayet “Notebook” içinde her şeyi yapmak isterseniz de “03-src” minimal kalabilir. Ama gerçek projelerde genelde fonksiyonları script’e koyup Notebook’ta import edilerek çağırılır.
-   **04-logs/**: Any log files (if using logging).
- Eğer “logging” kullanıyorsanız, log dosyaları burada toplanır (ör. info.log, error.log).
.gitignore ekleyebilirsiniz, loglar anlık üretildiği için Git’te tutmak istemeyebilirsiniz.
Gelişmiş projelerde logging modülü, verinin ingestion’ında bir pipeline log’ı tutar.
-   **05-docs/**: Additional documentation (architecture, data dictionary, references).
- “design_architecture.md”: “Proje veri akışı, ingestion pipeline, DB vs.”
“data_dictionary.md”: “Her kolonun anlamı, param birimleri vs.”
references/: Makale PDF’ler, external link notları, vb.
Bu klasör ekip veya ileride projeyi devralacak kişilerin dokümantasyon bulacağı yer.

.gitignore: data/raw/*, logs/*, vs. ignore.
README.md:
Projenin özet açıklaması, adım adım kurulum, “Nasıl çalıştırırım?”
Gerekirse tablo:
“Directory: 01-data, Purpose: ham/işlenmiş data.”
requirements.txt (veya environment.yml)
Python kütüphanelerinizin sürümlerini listeler.
LICENSE: MIT, Apache 2.0, vs. (opsiyonel).
-   **.gitignore**: Ignore large data files, logs, etc.
-   **requirements.txt**: Python libraries and versioning.
-   **LICENSE**: If open-sourcing, specify license.

3\. Setup Instructions
----------------------

1.  **Clone Repository**:
 
  ```bash
    git clone https://github.com/yourusername/<PROJECT_NAME>.git
   ```
    

2. **Create Virtual Environment (Recommended)**:

  ```bash
    cd <PROJECT_NAME>
    python -m venv venv
   ```
    ```bash
    # Activate venv
    # Mac/Linux
    source venv/bin/activate

3. **Install Requirements**:

  ```bash
    pip install -r requirements.txt
```
4. **Add EPA Data**:
Place raw CSV files into 01-data/raw/ subfolders (like epa-co-2022/, epa-so2-2023/, etc.).

4\. Usage
---------

**Jupyter Notebooks**:

-   Open `02-notebooks/01_data_check.ipynb` to see shape/columns checks.
-   Next, run `02_data_ingestion.ipynb` to unify & generate final CSV (`epa_long.csv`).
-   Explore data in `03_eda_exploration.ipynb`.

**Python Scripts**:

-   `03-src/data_ingestion.py`: Programmatically ingest & merge data.
-   `03-src/data_preprocessing.py`: Additional cleaning (null handling, outlier removal).

5\. Additional Info
-------------------

-   **docs/** folder holds design docs, references, data dictionary, etc.
-   **logs/** folder can store debug logs, if you enable logging in scripts.