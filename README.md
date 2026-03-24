# Quantitative Analysis of North American Bank Equities: Risk, Returns, and Portfolio Optimization

## Overview

This project develops a comprehensive financial analytics framework to evaluate the performance, risk characteristics, and macroeconomic sensitivities of major North American banking equities.

The analysis integrates Python-based quantitative modeling with an interactive Power BI dashboard to provide insights into return dynamics, downside risk, and optimal portfolio construction.

---

## Objective

The project aims to:

- Analyze risk-adjusted performance across major bank stocks  
- Quantify downside risk using Value-at-Risk (VaR) and Expected Shortfall (ES)  
- Simulate future return distributions using Monte Carlo methods  
- Identify key macroeconomic drivers of bank returns  
- Construct optimal portfolios using mean-variance optimization  

---

## Assets Analyzed

### Canadian Banks
- Royal Bank of Canada (RY.TO)  
- Toronto-Dominion Bank (TD.TO)  
- Bank of Montreal (BMO.TO)  
- Bank of Nova Scotia (BNS.TO)  
- Canadian Imperial Bank of Commerce (CM.TO)  

### U.S. Banks
- JPMorgan Chase (JPM)  
- Bank of America (BAC)  
- Goldman Sachs (GS)  
- Morgan Stanley (MS)  

### Benchmarks
- S&P 500 (^GSPC)  
- TSX Composite (^GSPTSE)  

---

## Data

- Source: Yahoo Finance (market data), FRED (macroeconomic data)  
- Frequency: Daily and monthly returns  
- Period: 2014 – Present  

---

## Methodology

### 1. Performance Analysis
- Compound Annual Growth Rate (CAGR)  
- Annualized volatility  
- Sharpe, Sortino, and Calmar ratios  
- Maximum drawdown  

### 2. Tail Risk Modeling
- Historical Value-at-Risk (VaR 99%)  
- Expected Shortfall (ES 99%)  
- Return distribution analysis  

### 3. Monte Carlo Simulation
- 1-year return simulations  
- Probability of loss estimation  
- Distribution of potential outcomes  

### 4. Macroeconomic Sensitivity
- ElasticNet regression (regularized linear model)  
- Random Forest (permutation feature importance)  
- Key drivers: interest rates, inflation, bond yields  

### 5. Portfolio Optimization
- Mean-variance optimization  
- Efficient frontier construction  
- Maximum Sharpe ratio portfolio  

---

## Key Findings

- Canadian banks exhibit **stronger risk-adjusted performance** and lower volatility  
- U.S. banks deliver **higher returns but significantly higher tail risk**  
- Downside risk (Expected Shortfall) is more pronounced in U.S. institutions  
- Interest rates and inflation are the **primary macroeconomic drivers of bank returns**  
- Diversification across Canadian and U.S. banks improves portfolio efficiency  


