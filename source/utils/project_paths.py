import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ProjectPaths:
    raw_dir: Path
    processed_dir: Path
    interim_dir: Path
    archive_dir: Path
    metadata_dir: Path
    logs_dir: Path
    plots_dir: Path
    metadata_path: Path
    save_dir: Path

    @classmethod
    def from_config(cls, config: Dict[str, Any], file_type: Optional[str] = None) -> "ProjectPaths":
        paths_config = config.get("paths", {})
        required_keys = ["raw_dir", "processed_dir", "archive_dir", "metadata_dir", "logs_dir"]
        missing_keys = [k for k in required_keys if k not in paths_config]
        if missing_keys:
            raise KeyError(f"Missing required config keys in paths: {missing_keys}")

        raw_dir = Path(paths_config["raw_dir"]).resolve()
        processed_dir = Path(paths_config["processed_dir"]).resolve()
        interim_dir = Path(paths_config.get("interim_dir", "../interim")).resolve()
        archive_dir = Path(paths_config["archive_dir"]).resolve()
        metadata_dir = Path(paths_config["metadata_dir"]).resolve()
        logs_dir = Path(paths_config["logs_dir"]).resolve()
        plots_dir = Path(paths_config.get("plots_dir", "../plots")).resolve()
        save_dir = Path(paths_config.get("save_dir", "../save")).resolve()
        metadata_path = metadata_dir / "processed_files.json"

        if file_type and "file_specific_paths" in config:
            file_specific = config["file_specific_paths"].get(file_type, {})
            for key, path_str in file_specific.items():
                if key in {"raw_dir", "processed_dir", "interim_dir", "archive_dir", "metadata_dir", "logs_dir", "plots_dir", "save_dir"}:
                    if key == "raw_dir":
                        raw_dir = Path(path_str).resolve()
                    elif key == "processed_dir":
                        processed_dir = Path(path_str).resolve()
                    elif key == "interim_dir":
                        interim_dir = Path(path_str).resolve()
                    elif key == "archive_dir":
                        archive_dir = Path(path_str).resolve()
                    elif key == "metadata_dir":
                        metadata_dir = Path(path_str).resolve()
                    elif key == "logs_dir":
                        logs_dir = Path(path_str).resolve()
                    elif key == "plots_dir":
                        plots_dir = Path(path_str).resolve()
                    elif key == "save_dir":
                        save_dir = Path(path_str).resolve()  # Yeni eklenen save_dir
        return cls(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            interim_dir=interim_dir,
            archive_dir=archive_dir,
            metadata_dir=metadata_dir,
            logs_dir=logs_dir,
            plots_dir=plots_dir,
            metadata_path=metadata_path,
            save_dir=save_dir
        )

    def ensure_directories(self) -> None:
        logger = logging.getLogger(__name__)
        for dir_path in [self.raw_dir, self.processed_dir, self.interim_dir, self.archive_dir, self.metadata_dir,
                         self.logs_dir, self.plots_dir, self.save_dir]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory created or already exists: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {e}")
                raise