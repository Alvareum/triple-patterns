# Triples in dynamic networks  
### An algorithm searching for triple patterns in dynamic networks  
This repository contains three modules:
### 1. Graph Randomization  
This module generates random graphs based on calcium activity data.  
### 2. Triple Calculation  
This module performs triple search on a graph and retrieves information associated with behavioral markup.  
### 3. Chi-square Calculation  
This module performs a statistical evaluation of the significance of triple activity in relation to behavioral markup.  

# Installation 
You can install the library from source using Poetry. Run the following command in the root directory of the project:  
`poetry install`

# Usage  
To verify the library's functionality, you can run the following Jupyter notebooks::  
1. `graph_randomization.ipynb`  
2. `triple_calc.ipynb`  
3. `chi2_calc.ipynb`  

# Dependencies  
- [Pandas](https://github.com/pandas-dev/pandas)  
- [Numpy](https://numpy.org/)  
- [Scipy](https://scipy.org/)  
- [Numba](https://numba.pydata.org)  
- [Statsmodels](https://www.statsmodels.org/)
