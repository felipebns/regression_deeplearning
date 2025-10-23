import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tickers = ["AAPL","MSFT","AMZN","GOOGL","NVDA","META","SPY","QQQ"]

# Baixa preços ajustados de fechamento
data = yf.download(tickers, period="10y", interval="1d", auto_adjust=True)["Close"]
data = data.dropna(how="all")  # remove linhas totalmente vazias

# Calcula retornos diários
returns = data.pct_change().dropna()

# Matriz de correlação
corr = returns.corr()
print(corr)

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Correlation matrix (daily returns, 10y)")
plt.tight_layout()
plt.show()