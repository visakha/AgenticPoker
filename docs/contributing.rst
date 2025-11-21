Contributing
============

Thanks for your interest in contributing to **Agentic Poker**! 🎉

Development Setup
-----------------

1. Clone the repository::

    git clone https://github.com/yourusername/AgenticPoker.git
    cd AgenticPoker

2. Create a virtual environment::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate

3. Install dependencies::

    pip install -r requirements.txt
    cd ui/poker_component && npm install

Code Style
----------

We use **Black** for code formatting. Before submitting a PR:

Format your code::

    make format

Or manually::

    black engine/ agents/ tests/ analysis/ *.py

Check formatting without changes::

    make format-check

Type Checking
-------------

All Python code must have **type hints**. We use **mypy** for static type checking::

    make type-check

Fix any type errors before submitting.

Testing
-------

Run all tests::

    make test

Run only BDD tests::

    make test-bdd

Run only unit tests::

    pytest tests/

All tests must pass before merging.

Documentation
-------------

Update documentation when adding features::

    make docs

Docs are built with Sphinx and located in ``docs/``.

Pull Request Guidelines
------------------------

1. **Fork** the repository
2. **Create a branch**: ``git checkout -b feature/my-awesome-feature``
3. **Make changes** with clear, atomic commits
4. **Format code**: ``make format``
5. **Type check**: ``make type-check``
6. **Test**: ``make test``
7. **Update docs** if needed
8. **Submit PR** with a clear description

Commit Messages
---------------

Use clear, descriptive commit messages:

* ✅ ``Add all-in detection to hand evaluator``
* ✅ ``Fix pot calculation bug in multi-way all-ins``
* ❌ ``fix stuff``
* ❌ ``wip``

Adding AI Personalities
-----------------------

To add a new personality:

1. Edit ``config/tune_ai_players.yml``
2. Add entry with VPIP, PFR, Aggression Factor, Bluff Frequency
3. Test with ``python cli.py --games 100``
4. Verify stats with ``python analysis/stats_analyzer.py game_logs.jsonl``

Code of Conduct
---------------

* Be respectful and constructive
* Help others learn
* Have fun! This is a hobby project 🎮

Questions?
----------

Open an issue or start a discussion. We're friendly! 😊
