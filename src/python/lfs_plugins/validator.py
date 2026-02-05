# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin validation utilities."""

from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def validate_plugin(plugin_path: str | Path) -> list[str]:
    """Validate plugin structure and manifest. Returns list of errors (empty if valid)."""
    plugin_dir = Path(plugin_path)
    errors = []

    if not plugin_dir.exists():
        return [f"Plugin not found: {plugin_dir}"]

    manifest = plugin_dir / "pyproject.toml"
    if not manifest.exists():
        errors.append("Missing pyproject.toml")
    else:
        try:
            data = tomllib.loads(manifest.read_text())
            lf = data.get("tool", {}).get("lichtfeld", {})
            if not lf:
                errors.append("pyproject.toml: missing [tool.lichtfeld] section")
            project = data.get("project", {})
            for field in ("name", "version", "description"):
                if field not in project:
                    errors.append(f"pyproject.toml: missing project.{field}")
            for field in ("auto_start", "hot_reload"):
                if field not in lf:
                    errors.append(f"pyproject.toml: missing tool.lichtfeld.{field}")
        except Exception as e:
            errors.append(f"pyproject.toml: {e}")

    init_py = plugin_dir / "__init__.py"
    if not init_py.exists():
        errors.append("Missing __init__.py")
    else:
        content = init_py.read_text()
        if "def on_load" not in content:
            errors.append("Missing on_load() function")
        if "def on_unload" not in content:
            errors.append("Missing on_unload() function")

    return errors
