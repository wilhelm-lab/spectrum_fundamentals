# Spectrum Fundamentals

Spectrum Fundamentals is a package we use in our rescoring pipeline. It was created to be able to handle spectra from different files.

## Features

* Annotate Spectra.
* Modify peptide sequence to follow one of the recommended notations of ProForma.
* Generate features than can be used by percoltor for rescoring (spectral angle, cosine similarity, Pearson's correlation, Spearman's correlation, etc.).

## Installation

Install with:

```
pip install git+https://github.com/wilhelm-lab/spectrum_fundamentals
```
    
## Usage


### Annotation Pipeline

The annotation script can be found in the package in annotation/annotation.py.

- Install and import the function:

```
from fundamentals.annotation.annotation import annotate_spectra
```

- Apply the function on any given dataframe with peptides meta data, raw intensities and mz:

```
annotate_spectra(dataframe)
```