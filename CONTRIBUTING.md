# Contributing

Thanks for your interest in AlayaRenderer-Flash.

## Development setup

```bash
uv sync                      # creates .venv with the pinned dependencies and dev tools
uv run pre-commit install    # runs ruff check + ruff format on every commit
uv run pytest -q             # CPU tests; GPU tests are skipped without CUDA
```

## Guidelines

- **Bugs and questions**: open an issue with your GPU, driver, CUDA and PyTorch
  versions, the exact command, and the full error output.
- **Pull requests**: keep each PR to one change and say how you tested it. CI
  only runs `ruff format --check`, `ruff check` and `uv lock --check` (it does
  not install PyTorch), so run `uv run pytest -q` locally before opening a PR. If you change the model
  path, also run a GPU render (`tests/test_gpu_smoke.py`); if you change the live
  demo, include the `alayarender-flash serve ... doctor --json` output.
- **License**: contributions are accepted under the Apache License 2.0. Do not
  submit SuperTuxKart source code or patches.
- Do not commit model weights, datasets, logs, or machine-specific paths.
