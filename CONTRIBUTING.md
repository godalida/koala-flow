# Contributing to Koala Flow

First off, thanks for taking the time to contribute! 🎉

Koala Flow is built by Data Engineers, for Data Engineers. We value clean code, clear documentation, and pragmatic solutions.

## How Can I Contribute?

### 1. Reporting Bugs
Found a bug? Open an issue on GitHub!
*   **Use a clear title.**
*   **Describe the steps to reproduce.**
*   **Provide code snippets** or error logs.
*   **Mention your environment** (OS, Python version, `koala-flow` version).

### 2. Suggesting Enhancements
Have an idea? We love new adapters (LightGBM, CatBoost, etc.) or core features!
*   **Open an issue** first to discuss the design.
*   **Explain "Why"** this feature is needed.
*   **Mock up the API** if you can.

### 3. Pull Requests
Ready to code?
1.  **Fork the repo** and clone it locally.
2.  **Create a branch** for your feature: `git checkout -b feature/my-new-adapter`.
3.  **Install dev dependencies:**
    ```bash
    pip install -e ".[dev]"
    ```
4.  **Make your changes.**
5.  **Run tests:**
    ```bash
    pytest
    ```
6.  **Lint your code:** (We use `ruff` or `black`)
    ```bash
    ruff check src/
    ```
7.  **Push and open a PR!**

## Development Setup

We use standard Python tooling.

```bash
# Clone
git clone https://github.com/your-username/koala-flow.git
cd koala-flow

# Virtual Env
python -m venv venv
source venv/bin/activate

# Install in editable mode with dev tools
pip install -e ".[dev]"
```

## Adding a New Adapter

If you want to add support for a new model framework (e.g., CatBoost), follow these steps:

1.  Create a new class in `src/koala_flow/adapters.py` inheriting from `ModelAdapter`.
2.  Implement `load(self, path)`: It must return the loaded model object.
3.  Implement `predict(self, data)`: It must accept data (Narwhals/Pandas/Polars) and return a list/array of predictions.
4.  Add a test in `tests/test_koala_flow.py` mocking the library import (so tests pass without installing heavy deps).

## Code Style
*   **Type Hints:** Use type hints everywhere.
*   **Docstrings:** Google-style docstrings for classes and public methods.
*   **Logging:** Use `logger` instead of `print`.

## License
By contributing, you agree that your contributions will be licensed under its MIT License.
