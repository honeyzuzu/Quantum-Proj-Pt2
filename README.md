# Quantum-Project (Quantum Multi Period Asset Allocation Problem)

This project uses simulations to estimate the potential return and risk of a portfolio with a specified asset allocation over a 10-years. The problem is divided into the following steps, with sample Python code

---

### 1. Asset Class Performance Data
A typical investment portfolio consists of three main asset classes:
- **US Equities**: Represented by the *S&P 500 Index*.
- **International Equities**: Represented by the *MSCI ACWI ex US Index*.
- **Global Fixed Income**: Represented by the *Bloomberg Global Aggregate Index*.

We may use alternate investments in the future. 

---

### 2. Historical Portfolio Performance
We assume a portfolio with the following initial allocation:  
- **30% in US equities**  
- **30% in international equities**  
- **40% in global fixed income**

Starting from April 2004, the portfolio is rebalanced monthly.

---

### 3. Historical Risk Measures
We calculate key risk measures, including:
- **Covariance**
- **Correlation**
- **Volatility**

These metrics are derived from the logarithmic returns of the asset classes.

---

### 4. Performance Simulations
For the simulations, we:
- Use expected returns for the three asset classes rather than historical returns.
- Retain the historical covariance matrix, as it is more stable over time.

The simulation assumes that logarithmic returns follow a normal distribution. Over 10,000 iterations, we simulate 10 years of portfolio performance, obtaining the distribution of annual returns and volatilities.

<span style="color:red;">Using quantum computing could significantly accelerate simulations, especially for portfolios with more than three asset classes.</span>

---

### 6. Statistical Metrics
We evaluate the portfolio using the following metrics:
- **Sharpe Ratio**: The ratio of annualized return to annualized volatility. (For simplicity, we assume a risk-free rate of zero.)
- **Maximum Drawdown**: The largest peak-to-trough decline in portfolio value.
- **Calmar Ratio**: The ratio of annualized return to maximum drawdown.

