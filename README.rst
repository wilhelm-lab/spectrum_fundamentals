Spectrum Fundamentals
=====================

|PyPI| |Python Version| |License| |Read the Docs| |Build| |Tests| |Codecov| |pre-commit| |Black|

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
.. |Build| image:: https://github.com/wilhelm-lab/spectrum_fundamentals/workflows/Build%20spectrum_fundamentals%20Package/badge.svg
   :target: https://github.com/wilhelm-lab/spectrum_fundamentals/actions?workflow=Package
   :alt: Build Package Status
.. |Tests| image:: https://github.com/wilhelm-lab/spectrum_fundamentals/workflows/Run%20spectrum_fundamentals%20Tests/badge.svg
   :target: https://github.com/wilhelm-lab/spectrum_fundamentals/actions?workflow=Tests
   :alt: Run Tests Status
.. |Codecov| image:: https://codecov.io/gh/wilhelm-lab/spectrum_fundamentals/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/wilhelm-lab/spectrum_fundamentals
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black

spectrum_fundamentals is a package primarily developed for usage within the rescoring and spectral library generation pipeline oktoberfest (https://github.com/wilhelm-lab/oktoberfest).

It provides the following functionalities:
 -  conversion between search engine-specific modstrings and the ProForma standard
 -  calculation of theoretical peptide / ion masses
 -  annotation of spectra
 -  spectral similarity calculation with various metrics
