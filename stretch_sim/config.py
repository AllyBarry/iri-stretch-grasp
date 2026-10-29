from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_PKG = "stretch_floor_perception"


def config_dir() -> Path:
    '''
    The package's config directory: the installed share directory when running from a
    colcon install, otherwise the source tree (so the sim also runs straight from the repo).
    '''
    source = Path(__file__).resolve().parents[1] / "config"
    if source.is_dir():
        return source
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory(_PKG)) / "config"


def resolve(name_or_path: str, subdir: str) -> Path:
    '''A bare name like "apartment" resolves to config/sim/<subdir>/apartment.yaml.'''
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml", ".json") or path.is_absolute() or len(path.parts) > 1:
        return path
    return config_dir() / "sim" / subdir / f"{name_or_path}.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def default_priors_file() -> Path:
    return config_dir() / "locations" / "location_priors.json"


def load_scenario(name_or_path: str, seed: Optional[int] = None) -> Dict[str, Any]:
    scenario = load_yaml(resolve(name_or_path, "scenarios"))
    if seed is not None:
        scenario["seed"] = seed
    return scenario
