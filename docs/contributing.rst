Contributor Guide
=================

Thank you for your interest in improving this project.
This project is open-source under the `MIT license`_ and
highly welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- `Source Code`_
- `Documentation`_
- `Issue Tracker`_
- `Code of Conduct`_

.. _MIT license: https://opensource.org/licenses/MIT
.. _Source Code: https://github.com/wilhelm-lab/spectrum_fundamentals
.. _Documentation: https://spectrum-fundamentals.readthedocs.io/
.. _Issue Tracker: https://github.com/wilhelm-lab/spectrum_fundamentals/issues

How to report a bug
-------------------

Report bugs on the `Issue Tracker`_.


How to request a feature
------------------------

Request features on the `Issue Tracker`_.


How to set up your development environment
------------------------------------------

You need Python 3.10+ and Poetry_.

Install the package with all development requirements:

.. code:: console

   $ make install

Install the pre-commit hooks (runs automatically on ``git commit``):

.. code:: console

   $ pre-commit install

You can now run an interactive Python session or the CLI:

.. code:: console

   $ poetry run python
   $ poetry run spectrum_fundamentals

.. _Poetry: https://python-poetry.org/


How to test the project
-----------------------

Run the full test suite:

.. code:: console

   $ make test

Run tests with a coverage report:

.. code:: console

   $ make coverage

Unit tests are located in the ``tests`` directory
and are written using the pytest_ testing framework.

.. _pytest: https://pytest.readthedocs.io/


How to lint and format code
---------------------------

Auto-fix formatting and style issues:

.. code:: console

   $ make format

Check without modifying files (what CI runs):

.. code:: console

   $ make lint

Run static type checking:

.. code:: console

   $ make type-check


Run all CI checks locally
-------------------------

Before opening a pull request, run the same checks that CI will run:

.. code:: console

   $ make check

This is equivalent to ``make lint && make type-check && make test``.


How to build and view the documentation
---------------------------------------

Build the docs:

.. code:: console

   $ make docs

Build and serve with live reload:

.. code:: console

   $ make docs-serve

The generated HTML files are in ``docs/_build/html/``.

.. _sphinx: https://www.sphinx-doc.org/en/master/


How to submit changes
---------------------

Open a `pull request`_ to submit changes to this project against the ``development`` branch.

Your pull request needs to meet the following guidelines for acceptance:

- The CI test suite must pass without errors and warnings.
- Include unit tests. This project maintains a high code coverage.
- If your changes add functionality, update the documentation accordingly.

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

.. _pull request: https://github.com/wilhelm-lab/spectrum_fundamentals/pulls
.. _Code of Conduct: CODE_OF_CONDUCT.rst


How to make a release
---------------------

Releases are published to PyPI automatically when a GitHub Release is published.
The version string lives only in ``pyproject.toml`` — ``__version__`` is read from
the installed package metadata at runtime.

Release Drafter continuously updates a draft GitHub Release with an accumulated changelog
from merged PR labels and a suggested next version (e.g. ``0.9.1``). It is a changelog
generator — it never modifies any file in the repository.

**Branch model:** ``development`` is the integration branch; ``main`` is the release branch.
Every commit on ``main`` corresponds exactly to a published release.

1. **Check the draft release** on GitHub to see the suggested next version (e.g. ``0.10.0``).
   The version is inferred automatically from the labels on merged PRs since the last release.

2. **Bump the version on** ``development``:

   .. code:: console

      $ git checkout development && git pull
      $ poetry version <next-version>   # e.g. poetry version 0.10.0
      $ git add pyproject.toml
      $ git commit -m "bump version to $(poetry version -s)"
      $ git push origin development

3. **Open a pull request** ``development → main`` titled ``Release v<next-version>``.
   CI runs automatically on the PR. Merge only when all checks pass.

4. **Publish the draft release** on GitHub.
   Because ``commitish: main`` is set in ``.github/release-drafter.yml``, the draft
   already targets ``main``. Simply click **Publish release** — no manual branch selection
   is needed. This triggers the publish workflow, which:

   - Re-runs the full CI suite as a hard gate.
   - Builds the wheel and sdist with ``poetry build``.
   - Publishes to PyPI via OIDC Trusted Publishing (no secrets required).
   - Creates the tag ``v<next-version>`` on ``main``.

5. **Back-merge** ``main`` into ``development`` so that ``development`` contains the
   release tag commit and stays in sync:

   .. code:: console

      $ git checkout development && git pull
      $ git merge main --no-ff -m "sync main back to development after release"
      $ git push origin development
