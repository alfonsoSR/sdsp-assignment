# SDSP Assignment - Alfonso Sánchez Rodríguez

## Contents of the repository

This repository includes all the material I generated during the assignment, including source code, output files, figures and the report. The original matlab script and data file provided with the assignment are also included, as well as the PDFs with the statement and support slides.

## Running the code

Most of the functionality is implemented in a dedicated Python library called `dspy`. Running any of the scripts in `code` requires installing this library. You can do this by opening a command line in `code/dspy` and running `pip3 install .`. All the dependencies will be installed automatically. No additional dependencies are required in order to run the scripts.

There are four python scripts in `code`:

- `generate_image.py`: Compute image of the complete field using algorithm selected by the user and save results to `code/images/<method>_image.pkl`. Available methods are `dirty`, `mvdr` and `aar`.
- `cropped_image.py`: Compute high-resolution image cropped around brightest source using algorithm selected by the user and save results to `code/images/<method>_cropped_image.pkl`. Available methods are `dirty`, `mvdr` and `aar`.
- `show_image.py`: Produce a plot of the image generated with the method chosen by the user and save it as `report/figures/<method>.png`. Possible inputs are `dirty`, `mvdr`, `aar`, and `<method>_cropped` for the three mentioned methods.
- `scores.py`: Produce a plot with the image of the complete field on the left and the cropped image on the right, and print the scores of the algorithms. The figures are saved as `report/figures/<method>-dual.png`. Possible inputs are `dirty`, `mvdr`, and `aar`.
