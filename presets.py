"""Preset storage for the image splitter's global options.

A preset captures the global (non per-page, non-CBZ) settings so a user can
save and reuse configurations. Presets persist as JSON in the user's config
directory. A built-in "Default" preset always exists and cannot be deleted,
though its stored values may be overwritten; the original factory values are
always recoverable via :data:`ORIGINAL_DEFAULTS`.
"""

from __future__ import annotations

import json
import logging
import os

import image_utils as iu

logger = logging.getLogger(__name__)

DEFAULT_PRESET_NAME = "Default"

# Factory defaults for the global options. These are the fallback values and
# the source for "reset to original defaults" even if the Default preset has
# been overwritten.
ORIGINAL_DEFAULTS: dict = {
    "desired_height": 0,
    "order": iu.RIGHT_TO_LEFT,
    "crop_mode": iu.CROP_GLOBAL,
    "threshold": 25,
    "skip_outer": 0,
    "skip_top": 0,
    "skip_bottom": 0,
    "detect_vertical": False,
    "detect_grayscale": False,
    "grayscale_tolerance": iu.GRAYSCALE_TOLERANCE,
}

# The settings keys a preset stores (order defines nothing functional).
SETTING_KEYS = tuple(ORIGINAL_DEFAULTS.keys())


def _config_path() -> str:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return os.path.join(base, "MangaTool", "presets.json")


def sanitize(settings: dict) -> dict:
    """Return a settings dict with only known keys, filling missing defaults.

    Types are coerced to match :data:`ORIGINAL_DEFAULTS` so values coming from
    JSON or the UI are normalized before use.
    """
    out: dict = {}
    for key, default in ORIGINAL_DEFAULTS.items():
        value = settings.get(key, default)
        try:
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, int):
                value = int(value)
            else:
                value = str(value)
        except (ValueError, TypeError):
            value = default
        out[key] = value
    return out


class PresetStore:
    """Loads, saves, and manages named presets backed by a JSON file."""

    def __init__(self, path: str | None = None):
        self.path = path or _config_path()
        self.presets: dict[str, dict] = {}
        self.load()

    # ------------------------------------------------------------- storage
    def load(self):
        data = {}
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.info(
                "No usable presets file at %s (%s); using defaults.",
                self.path, exc,
            )
            data = {}
        raw = data.get("presets", {}) if isinstance(data, dict) else {}
        self.presets = {
            str(name): sanitize(vals)
            for name, vals in raw.items()
            if isinstance(vals, dict)
        }
        # Guarantee the Default preset exists.
        if DEFAULT_PRESET_NAME not in self.presets:
            self.presets[DEFAULT_PRESET_NAME] = dict(ORIGINAL_DEFAULTS)
        logger.info(
            "Loaded %d preset(s) from %s.", len(self.presets), self.path
        )

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"presets": self.presets}, fh, indent=2)
        logger.info("Saved %d preset(s) to %s.", len(self.presets), self.path)

    # ------------------------------------------------------------- queries
    def names(self) -> list[str]:
        """Preset names, with Default first then the rest alphabetically."""
        others = sorted(n for n in self.presets if n != DEFAULT_PRESET_NAME)
        return [DEFAULT_PRESET_NAME] + others

    def get(self, name: str) -> dict:
        return dict(self.presets.get(name, ORIGINAL_DEFAULTS))

    def exists(self, name: str) -> bool:
        return name in self.presets

    @staticmethod
    def is_protected(name: str) -> bool:
        """Default preset may be overwritten but never deleted/renamed."""
        return name == DEFAULT_PRESET_NAME

    # ----------------------------------------------------------- mutations
    def set(self, name: str, settings: dict):
        self.presets[name] = sanitize(settings)
        self.save()

    def delete(self, name: str) -> bool:
        if self.is_protected(name) or name not in self.presets:
            return False
        del self.presets[name]
        self.save()
        return True

    def reset_default(self):
        """Restore the Default preset to the original factory values."""
        self.presets[DEFAULT_PRESET_NAME] = dict(ORIGINAL_DEFAULTS)
        self.save()
