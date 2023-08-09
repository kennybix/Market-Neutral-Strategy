# MNS-Intercept

<h1>SUMMARY</h1>
In this project, the market-neutral trading strategy is created using python. Several steps were taken to achieve this task. In summary, the steps taken are;
<ol type='1'>
<li> The data was adequately explored and cleaned.
<li> Rules were put in place to guide the pair selection process. First, DBSCAN clustering algorithm was used to cluster the returns of the stocks. The average volume of the stocks in the training period was used to improve the quality of the clustering process. Then an improved pair selection algorithm which involves multiple criteria (Hurst exponent, half life, zero crossing, cointegration etc) was used to select the best trading pairs.
<li> The allocations of the assets for each pair was done equally since information such as Beta is not available and I do not want to do very complex portfolio optimization for each of the pairs.
<li> The signals for each pair were generated using the z-score with the training data. The z-score in this solution used the 25th and 75th percentiles to compute the lower and upper thresholds for each pair. The strategy framework produced also made available the use of the traditional mean and standard deviation for threshold computation.
<li> Also, for the signal generation, feature selection optimization was introduced to help improve the robustness of the signals generated.
<li> On the test data, the optimal feature from the signal generation stage is used to predict the signal using a simple trained machine learning model. Backtesting was carried out and the results of the process are presented.

</ol>
