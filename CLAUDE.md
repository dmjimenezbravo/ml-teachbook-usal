# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Claude Code — Configuración Específica

- Siempre responde en español.
- Usa el `.venv` del proyecto para ejecutar cualquier script Python.
- Antes de ejecutar `python scripts/...`, verifica que `.venv` existe. Si no, sugiere ejecutar `python scripts/setup_env.py`.

## Tests

Los tests unitarios viven en `tests/` y usan `unittest` (stdlib). Requieren el entorno dev: `python scripts/setup_env.py --yes --dev`.

- Ejecutar toda la suite: `.venv/Scripts/python.exe -m unittest discover -s tests` (Windows) o `.venv/bin/python -m unittest discover -s tests` (macOS/Linux)
- Ejecutar un solo archivo: `.venv/Scripts/python.exe -m unittest tests.test_check_encoding`
- Ejecutar un solo caso: `.venv/Scripts/python.exe -m unittest tests.test_check_encoding.CheckEncodingTests.test_detects_invalid_utf8`

`tests/verify_languages.py` y `tests/verify_sidebar.py` son scripts Playwright (no `unittest`) que verifican el selector de idioma y la barra lateral contra una preview real: lanza primero `python scripts/launch_preview.py --background` y ejecútalos por separado, p. ej. `.venv/Scripts/python.exe tests/verify_languages.py`.

El workflow `Test Clean Setup` (`.github/workflows/test.yml`) prueba una instalación limpia en Windows/macOS pero es solo manual (`workflow_dispatch`); no lo lances como parte del flujo normal.
