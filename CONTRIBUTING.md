# Contributing

Create a focused branch (`feature/...`, `fix/...`, or `docs/...`) from the default branch. Use concise Conventional Commit messages. Keep pull requests small, explain behavior changes and risk, and include tests for modified behavior.

Before opening a pull request run:

```bash
ruff check backend scripts tests
black --check backend scripts tests
pytest
python -m scripts.smoke_test --local
```

Model changes must include regenerated registry metadata, actual metrics, dataset/artifact hashes, and updated model/data documentation. Never commit secrets, personal lifestyle records, invented scientific sources, or unverified accuracy claims. Security issues should be reported privately to the owner rather than as a public exploit report.
