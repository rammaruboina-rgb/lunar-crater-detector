# Contributing to Lunar Crater Detector

Thank you for your interest in contributing to the Lunar Crater Detector project! We welcome contributions from the community. Please follow the guidelines below.

## Code of Conduct

This project is committed to providing a welcoming and inspiring community environment. Please read and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the [issue list](https://github.com/rammaruboina-rgb/lunar-crater-detector/issues) as you might find out that you don't need to create one. When creating a bug report, please provide:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Screenshots or logs** if applicable
- **Your environment** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancements are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Clear description** of the enhancement
- **Rationale** for why this would be useful
- **Examples** of how it would work

### Pull Requests

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request with a clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/lunar-crater-detector.git
cd lunar-crater-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pylint black mypy pytest pytest-cov
```

## Code Quality

### Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

### Testing

```bash
# Run all tests
python -m pytest test_crater_detection.py -v

# Run with coverage
python -m pytest test_crater_detection.py --cov=crater_detector --cov-report=html
```

### Linting

```bash
# Lint with pylint
pylint crater_detector.py

# Format with black
black crater_detector.py

# Type check with mypy
mypy crater_detector.py
```

## Commit Messages

- Use clear and descriptive commit messages
- Use the imperative mood ("Add" instead of "Added")
- Reference issues and pull requests when relevant
- Example: `Fix #123: Improve edge detection algorithm`

## Documentation

- Update README.md if your changes affect usage
- Add docstrings to new functions/classes
- Update CHANGELOG.md with significant changes
- Include examples in code comments for complex logic

## Review Process

1. Code will be reviewed for functionality, style, and tests
2. Feedback will be provided if changes are needed
3. Once approved, the PR will be merged

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue or contact the maintainers if you have any questions.

Thank you for contributing!
