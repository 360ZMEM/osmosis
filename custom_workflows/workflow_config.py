from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


def load_workflow_config(config_spec: str | None) -> dict[str, Any]:
    if not config_spec:
        return {}

    loaded_config = _load_config_object(config_spec)
    plain_config = _to_plain_data(loaded_config)
    if plain_config is None:
        return {}
    if not isinstance(plain_config, dict):
        raise TypeError(f"Workflow config must resolve to a mapping, got {type(plain_config).__name__}.")
    return plain_config


def apply_config_overrides(target: Any, overrides: dict[str, Any]) -> Any:
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise AttributeError(f"Config object {type(target).__name__} has no attribute '{key}'.")

        current_value = getattr(target, key)
        if isinstance(value, dict) and _is_nested_config(current_value):
            apply_config_overrides(current_value, value)
        else:
            setattr(target, key, value)
    return target


def _is_nested_config(value: Any) -> bool:
    return is_dataclass(value) or hasattr(value, "__dict__")


def _load_config_object(config_spec: str) -> Any:
    config_path = Path(config_spec).expanduser()
    if config_path.exists():
        return _load_from_path(config_path)

    module_name, separator, attribute_name = config_spec.partition(":")
    if not separator:
        return _resolve_module_config(importlib.import_module(module_name))

    if module_name.endswith(".py"):
        module = _load_python_module_from_path(Path(module_name).expanduser())
    else:
        module = importlib.import_module(module_name)

    return _materialize_config_object(getattr(module, attribute_name))


def _load_from_path(config_path: Path) -> Any:
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML workflow configs.") from exc

        with config_path.open("r", encoding="utf-8") as file_handle:
            return yaml.safe_load(file_handle) or {}

    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle) or {}

    if suffix == ".py":
        return _resolve_module_config(_load_python_module_from_path(config_path))

    raise ValueError(f"Unsupported workflow config file type: {config_path.suffix}")


def _load_python_module_from_path(module_path: Path) -> ModuleType:
    module_name = f"workflow_config_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import workflow config from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_module_config(module: ModuleType) -> Any:
    for attribute_name in ("WORKFLOW_CONFIG", "workflow_config", "CONFIG", "config"):
        if hasattr(module, attribute_name):
            return _materialize_config_object(getattr(module, attribute_name))

    if hasattr(module, "get_workflow_config"):
        return _materialize_config_object(module.get_workflow_config())

    return module


def _materialize_config_object(config_object: Any) -> Any:
    if isinstance(config_object, type):
        return config_object()
    if callable(config_object) and not isinstance(config_object, ModuleType):
        try:
            return config_object()
        except TypeError:
            return config_object
    return config_object


def _to_plain_data(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return _to_plain_data(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_plain_data(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return {key: _to_plain_data(item) for key, item in vars(value).items() if not key.startswith("_")}
    return value
