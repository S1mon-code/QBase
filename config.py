"""QBase configuration loader."""
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_config = None


def get_config() -> dict:
    """Load and cache config from config.yaml."""
    global _config
    if _config is None:
        with open(_CONFIG_PATH) as f:
            _config = yaml.safe_load(f)
    return _config


def get_alphaforge_path() -> str:
    """Return expanded AlphaForge path."""
    return str(Path(get_config()["alphaforge"]["path"]).expanduser())


def get_data_dir() -> str:
    """Return expanded AlphaForge data directory."""
    return str(Path(get_config()["alphaforge"]["data_dir"]).expanduser())
