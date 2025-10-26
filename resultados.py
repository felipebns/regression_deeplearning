"""
Script para análise de dados financeiros com feature engineering e normalização automática.
Baixa dados históricos da Apple (AAPL), cria features técnicas avançadas e aplica normalização adaptativa.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import normaltest
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ==================== FUNÇÕES DE DOWNLOAD ====================

def download_stock_data(ticker: str = "AAPL", period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    """
    Baixa dados históricos de ações usando yfinance.
    
    Args:
        ticker: Símbolo da ação (ex: "AAPL", "GOOGL")
        period: Período de dados (ex: "10y", "5y", "1y")
        interval: Intervalo dos dados (ex: "1d", "1h")
    
    Returns:
        DataFrame com dados históricos
    """
    print(f"\n[1] BAIXANDO DADOS: {ticker} ({period})...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    print(f"Shape dos dados: {df.shape}")
    return df


# ==================== FUNÇÕES DE FEATURE ENGINEERING ====================

def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features baseadas em preço."""
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']
    df['High_Low_Ratio'] = df['High'] / df['Low']
    return df


def create_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de retornos percentuais."""
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_3d'] = df['Close'].pct_change(3)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    return df


def create_moving_average_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de médias móveis."""
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    return df


def create_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de volatilidade (desvio padrão móvel)."""
    df['Volatility_5'] = df['Close'].rolling(window=5).std()
    df['Volatility_10'] = df['Close'].rolling(window=10).std()
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    return df


def create_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features baseadas em volume."""
    df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
    return df


def create_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de Bandas de Bollinger."""
    df['BB_Middle'] = df['MA_20']
    df['BB_Upper'] = df['MA_20'] + 2 * df['Volatility_20']
    df['BB_Lower'] = df['MA_20'] - 2 * df['Volatility_20']
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    return df


def create_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Cria feature RSI (Relative Strength Index)."""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def create_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features MACD (Moving Average Convergence Divergence)."""
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    return df


def create_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de momentum."""
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    return df


def create_roc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features ROC (Rate of Change)."""
    df['ROC_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
    df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Cria coluna target: preço de fechamento do próximo dia."""
    df['Target'] = df['Close'].shift(-1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas as funções de feature engineering.
    
    Args:
        df: DataFrame com dados OHLCV originais
    
    Returns:
        DataFrame com features criadas
    """
    print("\n[2] CRIANDO FEATURES AVANÇADAS...")
    
    df = create_price_features(df)
    df = create_return_features(df)
    df = create_moving_average_features(df)
    df = create_volatility_features(df)
    df = create_volume_features(df)
    df = create_bollinger_bands(df)
    df = create_rsi(df)
    df = create_macd(df)
    df = create_momentum_features(df)
    df = create_roc_features(df)
    df = create_target(df)
    
    print(f"Shape final após feature engineering: {df.shape}")
    print(f"Total de features criadas: {df.shape[1] - 5}")  # -5 para OHLCV originais
    print(f"Valores faltantes por coluna:\n{df.isnull().sum()}\n")
    
    return df


# ==================== FUNÇÕES DE VISUALIZAÇÃO ====================

def plot_feature_histograms(df: pd.DataFrame, n_cols: int = 4):
    """
    Plota histogramas de todas as features numéricas em grid.
    
    Args:
        df: DataFrame com features
        n_cols: Número de colunas no grid
    """
    print("\n[3] GERANDO HISTOGRAMAS...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_features = len(numeric_cols)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        data = df[col].dropna()
        
        if data.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(col)
            ax.set_axis_off()
            continue
        
        sns.histplot(data, bins=50, kde=True, ax=ax, stat='density', color='tab:blue')
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylabel('Density')
    
    # Desativa eixos extras se houver
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_scaling_comparison(df_original: pd.DataFrame, df_scaled: pd.DataFrame, 
                            applied: dict, max_features: int = 6):
    """
    Plota comparação entre features originais e normalizadas.
    
    Args:
        df_original: DataFrame original
        df_scaled: DataFrame normalizado
        applied: Dicionário com métodos de normalização aplicados
        max_features: Número máximo de features para plotar
    """
    print("\n[4] GERANDO COMPARAÇÃO DE NORMALIZAÇÃO...")
    
    cols = list(applied.keys())[:max_features]
    fig, axes = plt.subplots(len(cols), 2, figsize=(10, 3 * len(cols)))
    
    for i, col in enumerate(cols):
        # Original
        sns.histplot(df_original[col].dropna(), bins=50, kde=True, 
                    ax=axes[i, 0], color='tab:blue')
        axes[i, 0].set_title(f"{col} (original)")
        
        # Normalizado
        sns.histplot(df_scaled[col].dropna(), bins=50, kde=True, 
                    ax=axes[i, 1], color='tab:orange')
        axes[i, 1].set_title(f"{col} (scaled: {applied[col]})")
    
    plt.tight_layout()
    plt.show()


# ==================== FUNÇÕES DE NORMALIZAÇÃO ====================

def auto_scale_by_normality(df: pd.DataFrame, train_ratio: float = 0.8, alpha: float = 0.05, 
                           target_col: str = 'Target', save_scalers: bool = False, 
                           scalers_path: str = 'scalers') -> tuple:
    """
    Aplica normalização adaptativa baseada no teste de normalidade D'Agostino-Pearson.
    
    Para cada coluna numérica:
      - Aplica teste de normalidade nos dados de treino (prefixo cronológico)
      - Se p-value >= alpha → StandardScaler (z-score)
      - Caso contrário → MinMaxScaler (0-1)
    
    Args:
        df: DataFrame com features
        train_ratio: Proporção de dados para treino
        alpha: Nível de significância para teste de normalidade
        target_col: Nome da coluna target
        save_scalers: Se True, salva scalers em disco
        scalers_path: Diretório para salvar scalers
    
    Returns:
        Tupla (df_scaled, scalers, applied):
            - df_scaled: DataFrame normalizado (sem NaNs)
            - scalers: Dicionário com objetos scaler por coluna
            - applied: Dicionário com método aplicado por coluna
    """
    print("\n[5] APLICANDO NORMALIZAÇÃO ADAPTATIVA...")
    
    df_clean = df.copy().dropna()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target da lista de features, mas mantém para normalização
    include_target = target_col in numeric_cols
    if include_target:
        numeric_cols.remove(target_col)
    
    n = len(df_clean)
    train_len = max(int(n * train_ratio), 1)
    
    scalers = {}
    applied = {}
    
    for col in numeric_cols + ([target_col] if include_target else []):
        vals = df_clean[col].values.reshape(-1, 1)
        
        # Se poucos exemplos no treino, força MinMax
        if train_len < 8:
            scaler = MinMaxScaler(feature_range=(0, 1))
            choice = 'minmax'
            scaler.fit(vals)
        else:
            try:
                stat, p = normaltest(vals[:train_len].ravel())
            except Exception:
                # Se o teste falhar, assume não-normal
                p = 0.0
            
            if p >= alpha:
                scaler = StandardScaler()
                choice = 'zscore'
                scaler.fit(vals[:train_len])
            else:
                scaler = MinMaxScaler(feature_range=(0, 1))
                choice = 'minmax'
                scaler.fit(vals[:train_len])
        
        df_clean[col] = scaler.transform(vals).ravel()
        scalers[col] = scaler
        applied[col] = choice
    
    # Salva scalers se solicitado
    if save_scalers:
        os.makedirs(scalers_path, exist_ok=True)
        for k, sc in scalers.items():
            joblib.dump(sc, os.path.join(scalers_path, f'scaler_{k}.pkl'))
        print(f"Scalers salvos em: {scalers_path}/")
    
    return df_clean, scalers, applied


# ==================== EXECUÇÃO PRINCIPAL ====================

def main():
    """Função principal que executa o pipeline completo."""
    
    # 1. Download dos dados
    df = download_stock_data(ticker="AAPL", period="10y", interval="1d")
    
    # 2. Feature engineering
    df = engineer_features(df)
    
    # 3. Visualização das features originais
    plot_feature_histograms(df)
    
    # 4. Normalização adaptativa
    df_scaled, scalers, applied = auto_scale_by_normality(
        df, 
        train_ratio=0.8, 
        alpha=0.05, 
        target_col='Target', 
        save_scalers=True
    )
    
    # 5. Exibe métodos de normalização aplicados
    print("\nMétodos de normalização escolhidos (amostra):")
    for k, v in list(applied.items())[:20]:
        print(f"  {k}: {v}")
    
    # 6. Visualização comparativa (original vs normalizado)
    plot_scaling_comparison(df, df_scaled, applied, max_features=6)
    
    print("\n✓ Pipeline concluído com sucesso!")
    
    return df, df_scaled, scalers, applied


if __name__ == "__main__":
    df, df_scaled, scalers, applied = main()