plt.figure(figsize=(10, 6))
ax = sns.histplot(df['Volume'], bins=100, kde=True)
from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1e6:.0f}'))  # mostra em milhões (inteiros)
plt.title('Histograma dos Volumes negociados da AAPL')
plt.xlabel('Volume negociado (em milhões)')
plt.ylabel('Frequência')
plt.show()