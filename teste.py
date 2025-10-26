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



oaosdoadmoaismd

alpha = 0.02

scalers = {}
applied = {}

df_scaled = df.sample(n=1000, random_state=42).copy()

for col in df.columns:
    # usar os valores da amostra (df_scaled) para evitar mismatch de tamanho
    vals = df_scaled[col].values.reshape(-1, 1)
    vals = vals.astype('float')

    stat, p = normaltest(vals[0:100].ravel())  # faz com apenas uma parcela dos dados para evitar sensibilidade do teste
    print(f"{col}:{p}")

    if p >= alpha:
        scaler = StandardScaler()
        choice = 'zscore'
        scaler.fit(vals)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        choice = 'minmax'
        scaler.fit(vals)

    df_scaled[col] = scaler.transform(vals).ravel()

    scalers[col] = scaler
    applied[col] = choice

# print("Normality-based scaler choices (sample):")
# for k, v in list(applied.items())[:20]:
#     print(f"  {k}: {v}")

cols = list(applied.keys())

n_features = len(cols)
fig, axes = plt.subplots(n_features, 2, figsize=(12, 3 * n_features))
axes = np.atleast_2d(axes)

for i, c in enumerate(cols):
    ax_orig = axes[i, 0]
    ax_scaled = axes[i, 1]

    data_orig = df[c].dropna()
    data_scaled = df_scaled[c].dropna()

    sns.histplot(data_orig, bins=50, kde=True, ax=ax_orig, stat='density', color='tab:blue')
    ax_orig.set_title(f"{c} (original)")

    sns.histplot(data_scaled, bins=50, kde=True, ax=ax_scaled, stat='density', color='tab:orange')
    ax_scaled.set_title(f"{c} (scaled: {applied.get(c,'?')})")

plt.tight_layout()
plt.show()