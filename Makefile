.PHONY: help format format-check type-check test test-bdd docs clean install

help:
	@echo "🃏 Agentic Poker - Development Commands"
	@echo ""
	@echo "  make install      - Install all dependencies"
	@echo "  make format       - Format code with Black"
	@echo "  make format-check - Check formatting without changes"
	@echo "  make type-check   - Run mypy type checker"
	@echo "  make test         - Run all tests"
	@echo "  make test-bdd     - Run BDD tests only"
	@echo "  make docs         - Build Sphinx documentation"
	@echo "  make clean        - Remove build artifacts"
	@echo ""

install:
	pip install -r requirements.txt
	cd ui/poker_component && npm install

format:
	black engine/ agents/ tests/ analysis/ *.py

format-check:
	black --check engine/ agents/ tests/ analysis/ *.py

type-check:
	mypy engine/ agents/ analysis/ cli.py

test:
	pytest tests/
	behave tests/bdd/features/

test-bdd:
	behave tests/bdd/features/

docs:
	cd docs && sphinx-build -b html . _build/html
	@echo "📚 Docs built! Open docs/_build/html/index.html"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf docs/_build/
	rm -rf build/ dist/ *.egg-info
