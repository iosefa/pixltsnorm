# Contributing to pixltsnorm

Thank you for contributing to `pixltsnorm`!

## Code of Conduct

By participating in this project, please follow our [Code of Conduct](code_of_conduct.md).

## How Can I Contribute?

### Reporting Bugs

- **Check Existing Issues** — Before opening a new bug report, see if the issue has already been reported. If it has, add any additional details in a comment.

- **Submit a Report** — If the issue hasn't been reported, open a new issue and fill out the provided template.

### Suggesting Enhancements

Have an idea to improve pixltsnorm? Please open an issue to discuss your suggestion.

### Pull Requests

To contribute via pull requests:

1. **Fork the Repository** — Fork the pixltsnorm repository and clone it locally.
2. **Create a Branch** — Make changes in a new branch. Use descriptive names like `feat/`, `fix/`, or `docs/` followed by the feature or fix name.
3. **Commit Your Changes** — Write a clear commit message describing your changes.
4. **Push to Your Fork** — Push the branch to your fork on GitHub.
5. **Create a Pull Request** — Open a pull request (PR) in the pixltsnorm repository. Link any relevant issues.
6. **Code Review** — A maintainer will review your changes. You may need to make updates based on feedback.
7. **Merge** — Once approved, your PR will be merged into the main codebase.

## Style Guidelines

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to automate code formatting and checks. After cloning:

1. Install dev dependencies:
   ```bash
   pip install -e '.[dev]'
   ```

2. Install the pre-commit hooks:
   ```bash
   pre-commit install
   ```

Now, whenever you commit, Black and other checks run automatically. This ensures consistent style (Black formatting, no trailing whitespace, etc.) across the project with minimal manual effort.

### Python Code Style

- Follow the [PEP 8](https://pep8.org/) style guide.
- We use **[Black](https://black.readthedocs.io/en/stable/)** for code formatting:
  1. Install dev dependencies:
     ```bash
     pip install -e '.[dev]'
     ```
  2. Format your code:
     ```bash
     black .
     ```
  3. Confirm no changes are needed:
     ```bash
     black --check .
     ```
- Include type hints for function signatures.
- Add documentation to public APIs using docstrings.

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature").
- Limit the first line to 72 characters or fewer.
- Reference related issues and PRs when relevant.

## Releasing a New Version

### Steps for Creating a New Release

1. **Ensure `main` is up to date**:
   Confirm all changes intended for the release are merged into the `main` branch.

2. **Update the version**:
   Bump the version number based on the type of release (major, minor, or patch) following [semantic versioning](https://semver.org/) in `pyproject.toml`.
   - For example, if your current version is `0.1.0` and you are making a backward-compatible feature addition, you might change it to `0.2.0`.

3. **Create a new tag**:
   Tag the release with the new version using the format `vX.X.X`. For example:
   ```bash
   git tag v1.2.0
   git push origin v1.2.0
   ```

4. **Deploy to PyPI**:
   The GitHub Actions workflow will automatically build and deploy the package to PyPI once the tag is pushed.

### Semantic Versioning Guidelines
- **Major version**: For incompatible API changes.
- **Minor version**: For backward-compatible features.
- **Patch version**: For backward-compatible bug fixes.

## Additional Notes

- Ensure compatibility with the latest dependencies.
- Update documentation when adding or changing features.
