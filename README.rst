Spectrum Fundamentals
=====================

|PyPI| |Python Version| |License| |Read the Docs| |CI| |Codecov| |pre-commit| |Ruff|

.. |PyPI| image:: https://img.shields.io/pypi/v/spectrum_fundamentals.svg
   :target: https://pypi.org/project/spectrum_fundamentals/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/spectrum_fundamentals
   :target: https://pypi.org/project/spectrum_fundamentals
   :alt: Python Version
.. |License| image:: https://img.shields.io/github/license/wilhelm-lab/spectrum_fundamentals
   :target: https://opensource.org/licenses/MIT
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/spectrum_fundamentals/latest.svg?label=Read%20the%20Docs
   :target: https://spectrum-fundamentals.readthedocs.io/
   :alt: Read the documentation at https://spectrum-fundamentals.readthedocs.io/
.. |CI| image:: https://github.com/wilhelm-lab/spectrum_fundamentals/workflows/CI/badge.svg
   :target: https://github.com/wilhelm-lab/spectrum_fundamentals/actions?workflow=CI
   :alt: CI Status
.. |Codecov| image:: https://codecov.io/gh/wilhelm-lab/spectrum_fundamentals/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/wilhelm-lab/spectrum_fundamentals
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Ruff

spectrum_fundamentals is a package primarily developed for usage within the rescoring and spectral library generation pipeline oktoberfest (https://github.com/wilhelm-lab/oktoberfest).

It provides the following functionalities:
 -  conversion between search engine-specific modstrings and the ProForma standard
 -  calculation of theoretical peptide / ion masses
 -  annotation of spectra
 -  spectral similarity calculation with various metrics
