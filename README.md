# _Geo_Sentiment_Climate_ (Advanced Data Analytics Project)

## 1. Overview

Welcome to the **_Geo_Sentiment_Climate_** repository. This project aims to collect, clean, and analyze air quality
data (CO, SO2, NO2, O3, PM2.5, etc.) from the EPA (Environmental Protection Agency) and other sources. We'll unify the
datasets, handle missing values/outliers, and eventually perform advanced analytics/ML on them.

### Key Objectives:

1. **Data Ingestion & Preprocessing**: 
   - Gather raw CSV files from multiple states/parameters.
   - Standardize column names, unify param (co, so2, no2, pm25, o3).
2. **Exploratory Data Analysis**:
   - Visualize time series, geospatial distributions, etc.
3. **Feature Engineering & Modeling** (optional):
   - If time permits, build a forecast or regression model.
   - Evaluate model performance, visualize results.
4. **Evaluation & Visualization**:
5. **Documentation & Presentation**:

## 2. Repository Structure

```text
Geo_Sentiment_Climate/
  ├─ 00-config/
  │   ├─ settings.yml # Data source path, parameters, etc.
  │   ├─ credentials_template.yml
  │   
  ├─ 01-data/
  │   ├─ raw/
  │   │   └─ #  EPA CO, SO2, NO2, O3, PM2.5, etc.
  │   ├─ interim/
  │   │   └─ # Intermediate processed data, temporary files
  │   └─ processed/
  │   │   └─ # Cleaned/merged data (epa_long.csv, epa_wide.csv etc.)
  │   ├─ archive/
  │   │   └─ # Old data processing scripts or notes
  │   └─ metadata/
  │       └─ # Data dictionaries, schema information, etc.
  ├─ 02-notebooks/
  │   ├─ 01_data_check.ipynb
  │   ├─ 02_data_ingestion.ipynb
  │   ├─ 03_data_preprocessing.ipynb
  │   ├─ 04_missing_values.ipynb
  │   ├─ 05_eda_exploration.ipynb
  │   ├─ 06_feature_engineering.ipynb
  │   ├─ 07_modeling.ipynb
  │   └─ 08_evaluation_and_viz.ipynb
  ├─ 03-source/
  │   ├─ data_check.py
  │   ├─ data_ingestion.py
  │   ├─ data_preprocessing.py
  │   ├─ utils/
  │   │   └─ config_loader.py
  │       └─__init__.py
  │       └─logger.py
  │       └─common.py
  │       └─future_utils.py
  │       └─hash_utils.py
  │       └─metadata_manager.py
  │   └─ __init__.py
  ├─ 04-logs/
  │   └─ eda_exploration.log
  │   └─ data_ingestion.log
  │   └─ data_preprocessing.log
  │   └─ future_engineering.log
  │   └─ data_check.log
  │   └─ missing_values.log
  ├─ 05-docs/
  │   ├─ references/
  │   │   └─ # Article PDFs, external link notes, etc.
  │   ├─ design_architecture.md
  │   └─ data_dictionary.md
  ├─ 06-plots/
  ├─ .gitignore
  ├─ README.md
  ├─ requirements.txt
  └─ LICENSE 

```

### Directory Summary

-   **00-config/**: Project configuration (settings, credentials template).

Purpose: Store project configuration files (YAML, JSON, etc.) here.
settings.yml: For example, you can keep project configuration like "data source path, parameter list, etc." here.
credentials_template.yml: If you are using secret keys/tokens, put the "template" file here and add the actual
credentials to .gitignore.
This way, you apply a "config file" approach instead of hard-coded paths or parameters, leaving a professional
impression.

- **01-data/**: This folder structure is a common practice in data projects.
    -   `raw/`: Original CSV files from EPA, etc.
    -   `interim/`: Temporary intermediate files.
    -   `processed/`: Cleaned, standardized final CSVs (e.g. epa_long.csv).
    - `archive/`: Old data processing scripts or notes.
    - `metadata/`: Data dictionaries, schema information, etc.
-   **02-notebooks/**: Jupyter notebooks for data check, ingestion, EDA, modeling.
    - `01_data_check.ipynb`: CSV checks, shape/columns, mini-tests.
    - `02_data_ingestion.ipynb`: All ingestion/consolidation rename logic (or you can call src/data_ingestion.py).
    - `03_data_preprocessing.ipynb`: Null handling, outlier removal, etc.
    - `03_eda_exploration.ipynb`: Visual plots, correlation, anomaly detection.
    - `04_feature_engineering.ipynb`: Variable creation, unit conversion, demographic merge.
    - `05_modeling.ipynb`: Time series/ML model, param tuning.
    - `06_evaluation_and_viz.ipynb`: Result visualization, table, metrics.
    - `07_missing_values.ipynb`: Missing value imputation strategies.
      This way, notebooks go step-by-step, minimizing the "What did this file do?" question. You develop your code/cells
      in the relevant notebooks.
    - **03-source/**: Python scripts for ingestion, preprocessing, utilities, etc.
    - `data_check.py`: "Glob .csv" + shape check + reporting.
    - `data_ingestion.py`: rename_map, param dedect, concat + pivot.
    - `data_preprocessing.py`: null filling, outlier fix, units conversion, etc.
    - `eda_exploration.py`: Time series, geospatial, correlation, etc.
    - `feature_engineering.py`: New feature creation, demographic merge.
    - `missing_values.py`: Imputation strategies (mean, median, MICE).
    - `utils/`: helper functions (common.py, logger.py, etc.)
    - `__init__.py`: (Python package tradition, optional but professional.)
    - `logger.py`: (If you use logging, configure it here.)
    - `common.py`: Common functions (e.g. read/write CSV, timer, etc.)
    - `future_utils.py`: Future utility functions (e.g. new feature creation).
    - `hash_utils.py`: Hashing functions (e.g. for anonymization).
    - `metadata_manager.py`: Data dictionary, schema info, etc.
    - **04-logs/**: Any log files
    - `eda_exploration.log`: Log file for EDA exploration.
    - `data_ingestion.log`: Log file for data ingestion.
    - `data_preprocessing.log`: Log file for data preprocessing.
    - `future_engineering.log`: Log file for future engineering.
    - `data_check.log`: Log file for data check.
    - `missing_values.log`: Log file for missing value imputation.
    - **05-docs/**: Additional documentation (architecture, data dictionary, references).
    - `design_architecture.md`: "Project data flow, ingestion pipeline, DB, etc."
    - `data_dictionary.md`: "Meaning of each column, param units, etc."
    - `references/`: Article PDFs, external link notes, etc.
    - **06-plots/**: EDA plots, model performance charts, etc.
    - This folder is where the team or future project maintainers will find documentation.
    - **.gitignore**: Ignore large data files, logs, etc.
    - **requirements.txt**: Python libraries and versioning.
    - **LICENSE**: If open-sourcing, specify the license.

### Directory Summary

.gitignore: data/raw/*, logs/*, vs. ignore.
README.md:

“Directory: 01-data/raw/”
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
    git clone https://github.com/dogaaydinn/<Geo_Sentiment_Climate>.git
   ```
    

2. **Create Virtual Environment (Recommended)**:

  ```bash
    cd <Geo_Sentiment_Climate>
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
- Clean data in `03_data_preprocessing.ipynb`.
-   Explore data in `03_eda_exploration.ipynb`.
- Create new features in `04_feature_engineering.ipynb`.
- Impute missing values in `07_missing_values.ipynb`.
- Run `05_modeling.ipynb` for time series/ML model building.
- Evaluate results in `08_evaluation_and_viz.ipynb`.
- Run notebooks in order, or as needed.

**Python Scripts**:

-   `03-src/data_ingestion.py`: Programmatically ingest & merge data.
-   `03-src/data_preprocessing.py`: Additional cleaning (null handling, outlier removal).
- `03-src/eda_exploration.py`: Advanced EDA (time series, geospatial, correlation).
- `03-src/feature_engineering.py`: Create new features, merge demographic data.
- `03-src/missing_values.py`: Impute missing values (mean, median, MICE).
- `03-src/utils/common.py`: Common functions (read/write CSV, timer, etc.).
- `03-src/utils/logger.py`: Configure logging settings.
- `03-src/utils/future_utils.py`: Future utility functions.
- `03-src/utils/hash_utils.py`: Hashing functions.
- `03-src/utils/metadata_manager.py`: Data dictionary, schema info.
- Run scripts from the terminal or an IDE.

5\. Additional Info
-------------------

- **source/** folder contains Python scripts for data ingestion, preprocessing, etc.
- **.gitignore** excludes large files/logs from version control.
- **archives/** folder can store old scripts or notebooks.
- **tests/** folder can hold unit tests for your scripts.
- **interim/** folder can store temporary files (not pushed to Git).
- **processed/** folder can store cleaned/merged CSVs (pushed to Git).
- **metadata/** folder can store data dictionaries, schema info, etc.
- **raw/** folder can store original CSVs (not pushed to Git).
- **notebooks/** folder can store Jupyter notebooks for each step.
- **plots/** folder can store EDA plots, model performance charts.
- **requirements.txt** lists Python libraries for reproducibility.
- **docs/** folder holds design docs, references, data dictionary, etc.
- **logs/** folder can store debug logs, if you enable logging in scripts.
- **config/** folder can hold YAML/JSON files for settings, credentials, etc.

6\. Contribution
----------------

- **Fork** the repository.
- **Clone** the forked repository.
- **Create** a new branch.
- **Make** your changes.
- **Commit** your changes.

7\. License
----------------

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

8\. Contact
----------------

Maintainer: Doğa Aydın

Email: dogaa882@gmail.com

LinkedIn:[linkedin.com/in/dogaaydin](https://www.linkedin.com/in/dogaaydin/)

GitHub: [github.com/dogaa882](https://github.com/dogaaydinn)

Dev.to: [dev.to/dogaa882](https://dev.to/dogaa882)

Medium: [https://medium.com/@dogaa882](https://medium.com/@dogaa882)

