import ast
import yaml
import logging
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_headers(raw_dir: Path, output_csv: Path):

    header_rows = []

    csv_files = list(raw_dir.rglob("*.csv"))
    for csv_path in csv_files:
        file_name = csv_path.name
        try:
            df = pd.read_csv(csv_path, nrows=0)
            columns_list = df.columns.tolist()
            logger.info(f"Extracted headers from {file_name}")
        except Exception as e:
            columns_list = []
            logger.error(f"Could not read {file_name} due to {e}")

        header_rows.append({
            "file_name": file_name,
            "columns": columns_list
        })

    # Ensure the output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save the headers to a CSV file
    df_headers = pd.DataFrame(header_rows)
    # Save the columns as string
    df_headers.to_csv(output_csv, index=False)
    logger.info(f"Headers saved to {output_csv}")

def get_pollutant(file_name: str) -> str:

    fname = file_name.lower()
    if "so2" in fname:
        return "so2"
    elif "co" in fname:
        return "co"
    elif "no2" in fname:
        return "no2"
    elif "o3" in fname:
        return "o3"
    elif "pm2.5" in fname or "pm25" in fname:
        return "pm25"
    else:
        return "unknown"

def analyze_headers(headers_csv: Path):

    df_headers = pd.read_csv(headers_csv)
    logger.info(f"Loaded headers from {headers_csv}")
    # Convert the columns to list
    df_headers["columns_list"] = df_headers["columns"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
    )

    df_headers["pollutant"] = df_headers["file_name"].apply(get_pollutant)

    # Group by pollutant
    pollutant_groups = df_headers.groupby("pollutant")

    # Dictionary to store the results
    pollutant_columns = {}

    for pollutant, group in pollutant_groups:
        all_columns_sets = group["columns_list"].apply(set).tolist()
        # union of all columns (union):
        union_of_cols = set().union(*all_columns_sets)
        # intersection of all columns (intersection):
        intersection_of_cols = set(all_columns_sets[0])
        for s in all_columns_sets[1:]:
            intersection_of_cols = intersection_of_cols.intersection(s)

        logger.info(f"Pollutant: {pollutant}")
        logger.info(f"Number of files: {len(group)}")
        logger.info(f"ALL columns (union): {union_of_cols}")
        logger.info(f"COMMON columns (intersection): {intersection_of_cols}")
        logger.info("-----\n")

        pollutant_columns[pollutant] = {
            "all_columns": list(union_of_cols),
            "common_columns": list(intersection_of_cols)
        }

    # Load the existing settings.yml file
    settings_path = Path("../config/settings.yml")
    with open(settings_path, "r") as file:
        settings = yaml.safe_load(file)

    # Append the new column names to the settings
    if "columns" not in settings:
        settings["columns"] = {}
    settings["columns"].update(pollutant_columns)

    # Save the updated settings back to the file
    with open(settings_path, "w") as file:
        yaml.safe_dump(settings, file)

    logger.info("Settings updated successfully.")

if __name__ == "__main__":
    raw_dir = Path("../data/raw")
    output_csv = Path("../analysis/headers_extracted.csv")
    extract_headers(raw_dir, output_csv)
    analyze_headers(output_csv)