import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

def load_processed_files(metadata_path: Path) -> Dict[str, Any]:

    if not metadata_path.exists():
        return {"processed_files": []}
    try:
        with metadata_path.open("r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"processed_files": []}
    except Exception as e:
        logging.getLogger('metadata_manager').error(f"Error loading metadata: {e}")
        return {"processed_files": []}

def save_processed_files(data: Dict[str, Any], metadata_path: Path) -> None:

    logger = logging.getLogger('metadata_manager')
    logger.info(f"Saving processed files metadata to {metadata_path}")
    try:
        # Ensure the parent directory exists
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a temporary file path
        temp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")

        # Write to the temporary file first
        with temp_path.open("w") as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Temporary metadata file created at {temp_path}")

        # Replace the original file with the temporary file
        temp_path.replace(metadata_path)
        logger.info(f"Metadata file saved successfully at {metadata_path}")
    except Exception as e:
        logger.error(f"Error saving processed files metadata: {e}")
        raise

def is_file_processed(file_hash: str, metadata: Dict[str, Any]) -> bool:

    return any(file["file_hash"] == file_hash for file in metadata.get("processed_files", []))

def mark_file_as_processed(file_name: str, file_hash: str, rows_count: int, metadata: Dict[str, Any],
                           metadata_path: Path) -> None:

    metadata["processed_files"].append({
        "file_name": file_name,
        "file_hash": file_hash,
        "processed_at": datetime.now().isoformat(),
        "rows_count": rows_count
    })
    save_processed_files(metadata, metadata_path)
    logger = logging.getLogger('metadata_manager')
    logger.info(f"File {file_name} marked as processed with {rows_count} rows.")
