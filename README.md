# **Market-Neutral Trading Strategy (MNS-Intercept)**  

## **Project Overview**  
This project implements a **market-neutral trading strategy** focusing on **pairs trading** for Japanese stocks over the period **2013–2021**. By leveraging statistical arbitrage, machine learning, and a robust backtesting framework, the strategy systematically identifies and trades mean-reverting pairs.  

Key highlights include:  
- **DBSCAN clustering** with PCA for pair selection.  
- Rigorous pair selection using **statistical metrics** (cointegration tests, Hurst exponent).  
- **Signal generation** optimized using technical indicators and a **Random Forest Regressor**.  
- Backtesting framework evaluating performance through **CAGR, Sharpe Ratio, Maximum Drawdown**, and other key metrics.

---

## **Data Summary**  
- **Stock Universe**: 248 Japanese stocks.  
- **Time Horizon**:  
   - Training Set: **2013–2015**  
   - Test Set: **2016–2021**  
- **Data**: Daily price and volume data.  

---

## **Key Features**  

### **1. Data Preparation**  
- Removed stocks with >10% missing data and used **forward-filling** for imputation.  
- Ensured consistent date alignment, and removed duplicates to maintain data integrity.  

---

### **2. Pair Selection**  
The following steps were taken to identify robust trading pairs:  

#### **DBSCAN Clustering and t-SNE Visualization**  
- **PCA (Principal Component Analysis)** was applied to stock returns for dimensionality reduction.  
- Average trading volume was added as an additional clustering feature.  
- **DBSCAN** identified clusters of similar stocks based on returns and volume.  
- **t-SNE** was used to visualize the high-dimensional clustering results in **2D space**.  

**t-SNE Visualization Example**:  
![t-SNE Clustering](results/clustering_outcome/tsne_cluster_visualization.png)  

#### **Pair Selection Criteria**  
Pairs were selected based on the following quantitative thresholds:  
- **Minimum Zero Crossings**: ≥ 12 (ensuring spread stability).  
- **P-Value Threshold**: ≤ 0.05 (statistical cointegration).  
- **Hurst Exponent Threshold**: ≤ 0.5 (mean-reverting behavior).  

**Final Selected Pairs**:  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pairs</th>
      <th>t_statistic</th>
      <th>p_value</th>
      <th>coint_coef</th>
      <th>zero_cross</th>
      <th>half_life</th>
      <th>hurst_exponent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(6902 JT, 7203 JT)</td>
      <td>-2.954651</td>
      <td>0.039352</td>
      <td>0.594890</td>
      <td>39</td>
      <td>33</td>
      <td>0.430972</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(7911 JT, 9001 JT)</td>
      <td>-3.081635</td>
      <td>0.027942</td>
      <td>0.473723</td>
      <td>39</td>
      <td>27</td>
      <td>0.323993</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(7911 JT, 9005 JT)</td>
      <td>-3.051582</td>
      <td>0.030350</td>
      <td>0.713315</td>
      <td>47</td>
      <td>38</td>
      <td>0.447411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(7912 JT, 9021 JT)</td>
      <td>-3.176201</td>
      <td>0.021404</td>
      <td>0.174814</td>
      <td>36</td>
      <td>31</td>
      <td>0.402439</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(9001 JT, 9005 JT)</td>
      <td>-2.978553</td>
      <td>0.036946</td>
      <td>1.013671</td>
      <td>43</td>
      <td>38</td>
      <td>0.357287</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(9001 JT, 9007 JT)</td>
      <td>-3.793134</td>
      <td>0.002978</td>
      <td>0.663768</td>
      <td>69</td>
      <td>15</td>
      <td>0.254819</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(9001 JT, 9008 JT)</td>
      <td>-4.156469</td>
      <td>0.000780</td>
      <td>0.289039</td>
      <td>51</td>
      <td>15</td>
      <td>0.282388</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(9001 JT, 9009 JT)</td>
      <td>-3.555290</td>
      <td>0.006676</td>
      <td>0.287469</td>
      <td>39</td>
      <td>20</td>
      <td>0.308626</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(9001 JT, 9020 JT)</td>
      <td>-3.268528</td>
      <td>0.016345</td>
      <td>0.094821</td>
      <td>51</td>
      <td>23</td>
      <td>0.318699</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(9001 JT, 9021 JT)</td>
      <td>-3.669899</td>
      <td>0.004557</td>
      <td>0.094769</td>
      <td>39</td>
      <td>20</td>
      <td>0.310727</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(9001 JT, 9022 JT)</td>
      <td>-3.569096</td>
      <td>0.006380</td>
      <td>0.033434</td>
      <td>53</td>
      <td>19</td>
      <td>0.259801</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(9005 JT, 9021 JT)</td>
      <td>-3.109852</td>
      <td>0.025833</td>
      <td>0.135167</td>
      <td>41</td>
      <td>31</td>
      <td>0.390902</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(9009 JT, 9022 JT)</td>
      <td>-3.102209</td>
      <td>0.026390</td>
      <td>8.263229</td>
      <td>30</td>
      <td>29</td>
      <td>0.401706</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(9020 JT, 9021 JT)</td>
      <td>-3.958953</td>
      <td>0.001641</td>
      <td>1.001917</td>
      <td>60</td>
      <td>21</td>
      <td>0.367362</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(8331 JT, 8355 JT)</td>
      <td>-3.414943</td>
      <td>0.010460</td>
      <td>1.029639</td>
      <td>56</td>
      <td>22</td>
      <td>0.294103</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(8630 JT, 8766 JT)</td>
      <td>-3.718341</td>
      <td>0.003862</td>
      <td>0.942843</td>
      <td>62</td>
      <td>16</td>
      <td>0.254571</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(8725 JT, 8766 JT)</td>
      <td>-3.019312</td>
      <td>0.033130</td>
      <td>0.719093</td>
      <td>45</td>
      <td>23</td>
      <td>0.309631</td>
    </tr>
  </tbody>
</table>

---

### **3. Signal Generation**  
Signals for each trading pair were generated using:  

1. **Technical Indicators**:  
   - Ratios of **SMA** (5, 10, 30, 50), **EMA** (5, 10, 30, 50, 95), and **RSI (14)** for each pair.  
   - Calculated Z-scores to measure deviations from the mean.  


2. **Machine Learning with Random Forest**:  
   - Trained a **Random Forest Regressor** on the training data (2013–2015) to predict Z-scores on the test set (2016–2021).  
   - Predicted Z-scores were used to identify long/short positions:  
     - **Short**: If Z-score > upper threshold (75th percentile).  
     - **Long**: If Z-score < lower threshold (25th percentile).  

This integration of machine learning adds robustness and adaptability to the signal generation process.

---

### **4. Backtesting Framework**  
- Simulated portfolio performance with **equal capital allocation** across selected pairs.  
- Calculated the following metrics for both individual pairs and the aggregated portfolio:  
   - **Compound Annual Growth Rate (CAGR)**  
   - **Sharpe Ratio**  
   - **Maximum Drawdown (MDD)**  
   - **Annualized Volatility**  
   - **Sortino Ratio**  

---

## **Results**  

### **Clustering Visualization**  
- DBSCAN clustering with t-SNE visualization:  
![Cluster Visualization](results/clustering_outcome/tsne_cluster_visualization.png)  

### **Pair Trading Signals**  
- Z-score-based trading signals for selected pairs:  
![Trading Signals](results/signal/asset1_asset2_signal_chart.png)  

### **Portfolio Performance**  
- Test set cumulative portfolio returns:  
![Portfolio Performance](results/backtesting/asset1_asset2_portfolio_perf_chart.png)  

---

## **Overall Performance Metrics**  

| **Metric**                        | **Value**          |  
|----------------------------------|--------------------|  
| **Final Portfolio Value**         | $1,097,596.52      |  
| **Compound Annual Growth Rate**   | **1.81%**          |  
| **Maximum Drawdown (MDD)**        | **-4.34%**         |  
| **Annualized Volatility**         | **2.60%**          |  
| **Sharpe Ratio**                  | **0.70**           |  
| **Sortino Ratio**                 | **1.02**           |  

---
