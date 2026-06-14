# Contributing to django-neural-feed

First off, thank you for considering contributing to django-neural-feed! It's people like you who make django-neural-feed a great library for the community.

## 1. Branching Strategy

django-neural-feed follows a standard development workflow with `dev` as the main integration branch and `main` as the stable release branch:

- **`dev`**: The active development branch. All new features, optimizations, and standard bug fixes should target this branch.
- **`main`**: Production-ready code. Stable releases and PyPI tags are cut exclusively from here.

### How to cut your branches:
* **Features, Improvements, and Bug Fixes:** Always branch from `dev` and open your Pull Request against `dev`.
* **Hotfixes:** Only branch from `main` if you are fixing a critical production vulnerability that cannot wait for the next release cycle.

## 2. Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/django-neural-feed.git
   cd django-neural-feed
   ```
2. Create your feature branch: `git checkout -b feature/your-feature-name`
3. Install the package with development tools:
   ```bash
   pip install -e ".[dev]"
   ```
4. Database for Testing:
This package relies heavily on PostgreSQL and the pgvector extension. The easiest way to run a local instance is via Docker:

```bash
docker run --name dnf-postgres -e POSTGRES_DB=django_neural_feed_test_db -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d pgvector/pgvector:pg16
```

## 3. Code Style

We enforce standard Python formatting:
- Use `black` for code formatting.
- Use `ruff` for linting.
- Use `pytest` for testing.

Before submitting a Pull Request, ensure your code passes all local checks and includes relevant test cases for new functionality.

Before submitting a PR, run:
```bash
black .
ruff check .
pytest --cov=src/django_neural_feed
```

## 4. Pull Request Process

1. Keep each PR focused on one bug, feature, or documentation improvement.
2. Ensure your code passes the relevant checks before pushing.
3. Update `README.md`, if you added a new feature.
4. Use the provided PR template and link the related issue with `Fixes #<issue-number>` when applicable.
5. Respond to maintainer feedback with a follow-up commit instead of opening a duplicate PR.
