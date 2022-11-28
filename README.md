# GitHub BIO-210 project team6 6 Hopfield Network
The goal of this project is to simulate a neural network that evolves in time using the Hopfield network.
## Requirements
- python >= 3.10.6
- numpy
- matplotlib
- pytest
- os
## Installation guide
**To install numpy:**
`pip install numpy`
There might be issues with MacOS in installing numpy, if so consult: >https://phoenixnap.com/kb/install-numpy
**To install matplotlib :**
`pip install matplotlib`
**To install pytest:**
`pip install -U pytest`
**To install pip:**
>https://pip.pypa.io/en/stable/installation/
## Code organisation
- **`main.py`:** where the functions are called, where the project is run
- **`functions.py`:** where the functions are written
- **`functions_test.py` :** where the functions are tested
## Necessairy modifications to the code
In `main.py` line 82, the path must be changed to the directory of each person
## Running tests
### pytest
In terminal and in directory:
`pytest'
If tests do not run, or there are errors (for debugging):
`pytest --full-trace`
### doctest
In `test_functions.py`, `test_functions_doctest` will run the tests for doctests => testting of doctests in pytest
## How to run code
1. run `functions.py`
2. run `test_functions.py`
3. run `main.py`
4. run pytest



