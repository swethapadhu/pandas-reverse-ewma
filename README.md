### PANDAS REVERSE EWMA

The implementation of reverse exponential weighted moving average in pandas for retrieving the original values from ewma values.

Pandas has a method to calculate ewma for a given dataframe. This project adds a reverse funcionality to it by implementing a wrapper class for the same.

EwmWrapper.py - contains the wrapper class with forward and reverse ewma functionality.
main.py - imports this class and tests it on a sample dataframe, and prints the mse of the reversed and the original values.

Run:
python main.py
