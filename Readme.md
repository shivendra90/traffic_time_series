# Complete Time-Series Analysis on Traffic Congestion

This is a complete time-series workflow done in Python for a univariate response. There are a total of 63 features out of which several were manually engineered and contain collinearity that I have treated by simply dropping them and then comparing predictive performance.

This was primarily focussed on including only time-series algorithms but since gradient boosted and some linear methods too have been known to perform good on timestamp data, I therefore decided to perorm a comparative study for both categories.

The data used is basically competitive data hosted by IITM earlier this year, in the month of July. I've only extended my analysis on traffic congestion after my participation in the hackathon.

TODO:

- [x] Perform cleaning procedures.
- [x] Perform comparison of supervised learning methods.
- [x] Find out d, p, q terms for ARIMA models.
- [x] Implement time-series algorithms (SARIMAX).
- [ ] Implement a LSTM model.
- [x] Compare time-series methods with regression/supervised methods.
- [x] Add a fully functional jupyter notebook.
- [ ] Debug LSTM model(s) for full functionality.
