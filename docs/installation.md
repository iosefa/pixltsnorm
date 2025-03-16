# Installation

This document outlines how to install **pixltsnorm** from PyPI, GitHub, or via a Docker container.

---

## Prerequisites

- **Python 3.10+** (tested up to Python 3.13).

---

## 1. Install via PyPI

The simplest way to install pixltsnorm is from PyPI:

```bash
pip install pixltsnorm
```

This will install all necessary Python dependencies automatically (e.g., NumPy, scikit-learn). After installation, you can verify:

```bash
pip show pixltsnorm
```

Note, to use the `earth-engine` modules, you will need to separately install ...

---

## 2. Install via GitHub (Latest Development Version)

If you want the latest development code (possibly with unreleased features/fixes), install directly from the GitHub repository:

```bash
pip install git+https://github.com/iosefa/pixltsnorm.git
```

This approach clones the `main` branch of the repository and then installs it into your current environment.

---

## 3. Docker

An official or sample **Dockerfile** can be used to create a containerized environment with pixltsnorm preinstalled—useful if you want a standardized setup (e.g., for reproducible pipelines).

### Example Usage

If you have a Docker image (e.g. `iosefa/pixltsnorm:latest`) hosted on Docker Hub:

```bash
docker run -it --rm -p 8888:8888 iosefa/pixltsnorm:latest
```

This container might launch JupyterLab or a similar environment so you can run example notebooks demonstrating pixltsnorm usage.

---

## Verification & Usage

Once installed, you can test:

```bash
python -c "import pixltsnorm; print(pixltsnorm.__version__)"
```

or open a Python shell / notebook and import:

```python
import pixltsnorm
print(pixltsnorm.__version__)
```

You’re ready to explore and harmonize multi-sensor time series!

---

## Additional Notes

- For Earth Engine–based workflows, be sure the [Google Earth Engine Python API](https://developers.google.com/earth-engine/python_install) is installed and authenticated separately.
- If you encounter installation or dependency issues, please open an [issue](https://github.com/iosefa/pixltsnorm/issues) or refer to the community for help.
