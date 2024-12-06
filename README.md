# Quantum-Project (Quantum Multi Period Asset Allocation Problem)

This project uses simulations to estimate the potential return and risk of a portfolio with a specified asset allocation over a 10-year horizon. The problem is divided into the following steps, with sample Python code

---

### 1. Asset Class Performance Data
A typical investment portfolio consists of three main asset classes:
- **US Equities**: Represented by the *S&P 500 Index*.
- **International Equities**: Represented by the *MSCI ACWI ex US Index*.
- **Global Fixed Income**: Represented by the *Bloomberg Global Aggregate Index*.

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

