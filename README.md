# CL-1 Lab Practicals

Python code for the two CL-1 lab subjects. Every script has run instructions at the top,
step-by-step comments, and theory + viva notes at the end.

Run any script with `python <file>.py` (or paste it into Jupyter).
Install: `pip install numpy pandas matplotlib scikit-learn openpyxl statsmodels requests`

## ML - Machine Learning (417525)
| No. | File | Topic |
|---|---|---|
| 1 | `ML/1_PCA_Wine.py` | PCA on Wine dataset |
| 2 | `ML/2_Regression_Uber.py` | Linear, Ridge, Lasso regression (Uber fares) |
| 3 | `ML/3_SVM_Digits.py` | SVM on handwritten digits |
| 4 | `ML/4_KMeans_Iris.py` | K-Means + elbow method (Iris) |
| 5 | `ML/5_RandomForest_Car.py` | Random Forest (car evaluation) |
| 6 | `ML/6_QLearning_TicTacToe.py` | Q-learning Tic-Tac-Toe |

## DMV - Data Modeling and Visualization
| No. | File | Topic |
|---|---|---|
| 1 | `DMV/1_Multi_Format_Sales.py` | Load CSV / Excel / JSON sales data |
| 2 | `DMV/2_Weather_API.py` | OpenWeatherMap API (set `OPENWEATHER_API_KEY`) |
| 3 | `DMV/3_Telecom_Churn_Cleaning.py` | Data cleaning and preparation |
| 4 | `DMV/4_RealEstate_Wrangling.py` | Data wrangling |
| 5 | `DMV/5_AQI_Visualization.py` | matplotlib visualisation (AQI) |
| 6 | `DMV/6_Retail_Sales_Aggregation.py` | Data aggregation |
| 7 | `DMV/7_Stock_TimeSeries.py` | Time series + ARIMA forecast |

## Datasets
Datasets live in https://github.com/MrNobody4869/datasets. Every script that reads a file has two options in its load step:

```python
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/<file>.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("<file>.csv")
```

Option A is active by default. To use a local file, comment out Option A and uncomment Option B.
(`ML/4_KMeans_Iris.py` uses scikit-learn's built-in Iris as Option A and a local `Iris.csv` as Option B.)

| Script | Dataset file |
|---|---|
| ML/1_PCA_Wine.py | `wine.csv` |
| ML/2_Regression_Uber.py | `uber.csv` |
| ML/5_RandomForest_Car.py | `car_evaluation.csv` |
| DMV/3_Telecom_Churn_Cleaning.py | `Telecom_Customer_Churn.csv` |
| DMV/4_RealEstate_Wrangling.py | `RealEstate_Prices.csv` |
| DMV/5_AQI_Visualization.py | `City_Air_Quality.csv` |
| DMV/6_Retail_Sales_Aggregation.py | `Retail_Sales_Data.csv` |
| DMV/7_Stock_TimeSeries.py | `Stock_Prices.csv` |

Other scripts need no file (built-in data, self-generated data, or an API).

Sources: Uber fares (Kaggle yasserh), MSFT stock prices (Kaggle jek1wantaufik), India city air quality (Kaggle rohanrao), Superstore sales (Kaggle vivek468, reduced to 5 columns), Wine (UCI/GeeksforGeeks), Car evaluation (UCI), Telco churn (IBM sample), California housing (ageron).
