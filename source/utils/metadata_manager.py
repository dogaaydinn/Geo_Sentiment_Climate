import json
import os
from datetime import datetime
from typing import Dict


def load_processed_files(metadata_path: str) -> Dict:
    if not os.path.exists(metadata_path):
        return {"processed_files": []}
    with open(metadata_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"processed_files": []}


def save_processed_files(data: Dict, metadata_path: str):
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    temp_path = metadata_path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(temp_path, metadata_path)


def is_file_processed(file_hash: str, metadata: Dict) -> bool:
    return any(file["file_hash"] == file_hash for file in metadata["processed_files"])


def mark_file_as_processed(file_name: str, file_hash: str, rows_count: int, metadata: Dict, metadata_path: str):
    metadata["processed_files"].append({
        "file_name": file_name,
        "file_hash": file_hash,
        "processed_at": datetime.now().isoformat(),
        "rows_count": rows_count
    })
    save_processed_files(metadata, metadata_path)
