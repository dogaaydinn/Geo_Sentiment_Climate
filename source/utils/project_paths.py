from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

@dataclass
class ProjectPaths:
    raw_dir: Path
    processed_dir: Path
    archive_dir: Path
    metadata_dir: Path
    logs_dir: Path
    plots_dir: Path
    metadata_path: Path


    @classmethod
    def from_config(cls, config: Dict[str, Any], file_type: Optional[str] = None) -> "ProjectPaths":
        paths_config = config.get("paths", {})
        required_keys = ["raw_dir", "processed_dir", "archive_dir", "metadata_dir", "logs_dir"]
        missing_keys = [k for k in required_keys if k not in paths_config]
        if missing_keys:
            raise KeyError(f"Missing required config keys in paths: {missing_keys}")

        raw_dir = Path(paths_config["raw_dir"]).resolve()
        processed_dir = Path(paths_config["processed_dir"]).resolve()
        archive_dir = Path(paths_config["archive_dir"]).resolve()
        metadata_dir = Path(paths_config["metadata_dir"]).resolve()
        logs_dir = Path(paths_config["logs_dir"]).resolve()
        plots_dir = Path(paths_config.get("plots_dir", "../plots")).resolve()
        metadata_path = Path(paths_config.get("metadata_path", "../metadata.json")).resolve()

        # Eğer file_type parametresi varsa:
        if file_type and "file_specific_paths" in config:
            file_specific = config["file_specific_paths"].get(file_type, {})
            for key, path_str in file_specific.items():
                if key in {"raw_dir", "processed_dir", "archive_dir", "metadata_dir", "logs_dir", "plots_dir"}:
                    # Bu örnekte güncelleme yapıyoruz:
                    if key == "raw_dir":
                        raw_dir = Path(path_str).resolve()
                    elif key == "processed_dir":
                        processed_dir = Path(path_str).resolve()
                    elif key == "archive_dir":
                        archive_dir = Path(path_str).resolve()
                    elif key == "metadata_dir":
                        metadata_dir = Path(path_str).resolve()
                    elif key == "logs_dir":
                        logs_dir = Path(path_str).resolve()
                    elif key == "plots_dir":
                        plots_dir = Path(path_str).resolve()
        return cls(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            archive_dir=archive_dir,
            metadata_dir=metadata_dir,
            logs_dir=logs_dir,
            plots_dir=plots_dir,
            metadata_path=metadata_path
        )
