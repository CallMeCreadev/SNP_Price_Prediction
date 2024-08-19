This is a work in progress (Mostly Abandoned).
Original idea was to take a list of features in the stock market and related price data to attempt to calculate future moving averages.

Tutorial price prediction is a simple file designed to get the idea of basic prediction.

Data formatter was designed to take in the prices of various commodities and stocks and normalize their variances compared to the 10, 50, 100 day moving averages.
The goal was to use the variance from the moving averages on various commodities to attempt to identify the future movement of the SNP500 moving average.

training-room is where the model is trained and predictions are made and scored.

stockripper takes all of the stock data and saves it into a .pkl file to keep API calls to a minimum.
Feature extraction is a sample file to call data from the .pkl data file created by stockripper.

