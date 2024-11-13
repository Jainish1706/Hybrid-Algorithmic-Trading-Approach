# Algorithmic Trading Strategy Using ARIMA and LSTM Models
## Project Overview
This project develops a hybrid algorithmic trading strategy for predicting stock price movements and generating trading signals, specifically targeting the Nifty 500 index. The strategy combines linear (ARIMA) and non-linear (LSTM) modeling techniques along with technical indicators, such as Exponential Moving Averages (EMA) and the Average Directional Index (ADX). The approach aims to capture both historical trends and complex patterns in stock prices, enhancing predictive accuracy and profitability.

## Authors
- Jigar Shah, Pandit Deendayal Energy University
- Gautam Makwana, Pandit Deendayal Energy University
- Jainish Shah, IISER Thiruvananthapuram

## Note:
Link for the NIFTY 500 dataset: [Link Text](https://drive.google.com/drive/folders/1HAJhcRMbj2B25_ubXYdegyu_404O8vRw?usp=drive_link)

## Key Features
**1.** **Data Collection and Processing:** Historical market data for the Nifty 500 index is sourced from Yahoo Finance, including daily, weekly, and monthly price and volume data.<br>
**2. Modeling:** 
  - **ARIMA Model:** Captures linear patterns in price data using autoregressive, differencing, and moving average terms.
  - **LSTM Model:** Enhances ARIMA forecasts by capturing non-linear relationships in residuals, using a deep learning architecture.**<br>
  
**3. Trading Signals:**
  - EMA and ADX indicators are used to identify potential buy/sell points.
  - EMA crossovers and ADX slope changes inform the timing of trades.**<br>
  
**4. Performance Evaluation:** The Extended Internal Rate of Return (XIRR) is used to measure profitability, factoring in the timing and magnitude of trades.<br>

## Methodology:
The hybrid strategy incorporates ARIMA and LSTM models for forecasting. After data preprocessing, ARIMA provides initial forecasts, while LSTM refines predictions by addressing non-linear patterns. This approach improves decision-making accuracy and trading signal reliability. The final model is evaluated through backtesting and XIRR calculations to ensure effective returns.

## Results
The integrated ARIMA-LSTM model demonstrated high accuracy in predicting price movements and generating reliable trading signals. By incorporating both linear and non-linear insights, the hybrid strategy achieved superior risk-adjusted returns compared to using traditional technical indicators alone.

