import pandas as pd
import numpy as np

from algorithm.EwmWrapper import EwmWrapper

if __name__=='__main__':

	# Create df and a ewm wrapper object
	df = pd.DataFrame({"A":[0,1,2,0,4], "B":[1,2,3,1,2]})
	obj = EwmWrapper(df,com=0.5,adjust=True)
	print("Input dataframe")
	print(df)

	# Calculate ewma
	ewma = obj.calc_ewma()
	print("Exponential Weighted Moving Averaged df")
	print(ewma)

	# Calculate inverse
	inversed_df = obj.inverse_ewma()
	print("Inversed EWMA df")
	print(inversed_df)

	# Calculate error
	error = np.square(inversed_df.values-df.values).mean()
	print(f"The mse is: \n {error}")