Air Passengers Time Series Forecasting with LSTM
This project implements a time series forecasting model using a Long Short-Term Memory (LSTM) neural network to predict the number of international airline passengers.
The dataset used is the classic AirPassengers dataset, covering monthly totals from 1949 to 1960.

Dataset
Source: AirPassengers.csv
Features- Month: Month of observation (1949-01 to 1960-12)
Passengers: Total number of airline passengers per month

Project Structure
Data Preprocessing:
Convert the 'Month' column to datetime.
Set 'Month' as index.
Normalize passenger counts using MinMaxScaler.
Create input sequences for LSTM (12 months as input to predict the next month).

Model Architecture:
LSTM layer with 64 units and ReLU activation
Dense output layer
Loss function: Mean Squared Error (MSE)
Optimizer: Adam

Training:
80% training and 20% testing split
100 epochs

Evaluation:
Compare actual vs predicted passengers.
Visualize results using Matplotlib.

How to Run
Install required packages: pip install pandas numpy matplotlib scikit-learn tensorflow

Place the AirPassengers.csv file in your working directory.

Run the script:
bash
Copy
Edit
python lstm_airpassengers_forecast.py
The script will:
Train the LSTM model
Predict on the test set
Plot actual vs predicted passengers

Results
The model learns the time series pattern quite well, showing close alignment between the predicted and actual number of passengers.
<img src="path_to_output_plot.png" alt="Prediction Plot" width="600">

Future Work
Extend predictions beyond 1960 (multi-step forecasting)
Try more complex architectures: stacked LSTM, Bidirectional LSTM
Experiment with different sequence lengths
Hyperparameter tuning
