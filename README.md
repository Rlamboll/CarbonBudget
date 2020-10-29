# Carbon Budget Calculator
Calculates remaining carbon budget given atmospheric simulation inputs. 
The workhorse script is `src/run_budget_calculator.py`
Expects to find data from MAGICC or FaIR simulations in the `InputData` folder in the 
standard format output for a pyam dataframe. Specification of how to read this data 
and options for running the code are all found at the beginning of 
`src/run_budget_calculator.py`.   