import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import random

# Definir o caminho do arquivo Excel
caminho_arquivo = r"D:\TCC\Base_Limpa.xlsx"

# Ler as abas IPCA e CDI em DataFrames separados
df_ipca = pd.read_excel(caminho_arquivo, sheet_name='IPCA')
df_cdi = pd.read_excel(caminho_arquivo, sheet_name='CDI')

def preprocess_dataframe(df):
    # Garantir que as colunas relevantes estao no formato adequado
    df['Data Ref'] = pd.to_datetime(df['Data Ref'])
    df['Vencimento'] = pd.to_datetime(df['Vencimento'])
    
    # Excluir linhas onde a data de vencimento e anterior a data de hoje
    df = df[df['Vencimento'] >= pd.Timestamp.today()]
    
    # Identificar os registros mais recentes para a Taxa Indicativa, Duration e Rating com base na Data Ref
    df_processed = df.loc[df.groupby('Ticker')['Data Ref'].idxmax(), ['Ticker', 'Taxa Indicativa', 'Duration', 'Rating', 'Data Ref']]
    
    return df_processed

# Processar os dataframes IPCA e CDI
df_ipca = preprocess_dataframe(df_ipca)
df_cdi = preprocess_dataframe(df_cdi)

# Combinar os DataFrames em um unico DataFrame
df_portfolio = df_ipca.copy()

# Definir o mapeamento de risco medio baseado no Rating
risk_mapping = {
    'A1': 0.03,
    'A2': 0.05,
    'A3': 0.07,
    'B3': 0.10,
    'C': 0.15,
    'F': 0.20
}

# Adicionar a coluna Risco Medio ao DataFrame
df_portfolio['Risco Medio'] = df_portfolio['Rating'].map(risk_mapping)

# Verificar se ha algum Rating que nao foi mapeado
if df_portfolio['Risco Medio'].isnull().any():
    print("Existem Ratings nao mapeados. Verifique o mapeamento de risco.")
    df_portfolio = df_portfolio.dropna(subset=['Risco Medio'])

# Funcao para gerar alocacoes aleatorias com peso maximo
def generate_random_weights(num_assets, max_weight=0.15):
    weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
    weights = np.minimum(weights, max_weight)
    weights /= weights.sum()
    return weights

# Funcao para selecionar os ativos
def selecionar_ativos(df, n=500):
    return df.sample(n=n, random_state=np.random.randint(0, 10000)).reset_index(drop=True)

# Parametros para otimizacao
lambda_reg = 1.2
target_duration = 4
tolerance = 0.5

# Variaveis para armazenar resultados
resultados = []

# Simular 1000 carteiras
for i in range(10000):
    # Selecionar 10 ativos aleatorios
    selected_assets = selecionar_ativos(df_portfolio, n=10)
    
    # Gerar alocacao inicial
    initial_weights = generate_random_weights(len(selected_assets), max_weight=0.15)
    
    # Calcular metricas do portfolio inicial
    portfolio_return_initial = np.dot(initial_weights, selected_assets['Taxa Indicativa'])
    portfolio_risk_initial = np.sqrt(np.dot(initial_weights.T, np.dot(np.diag(selected_assets['Risco Medio'].values**2), initial_weights)))
    portfolio_duration_initial = np.dot(initial_weights, selected_assets['Duration'])
    
    # Variaveis para otimizacao
    weights_var = cp.Variable(len(selected_assets))
    portfolio_risk = selected_assets['Risco Medio'].values @ weights_var
    portfolio_return = selected_assets['Taxa Indicativa'].values @ weights_var
    portfolio_duration = selected_assets['Duration'].values @ weights_var

    # Restricoes de otimizacao
    target_min_duration = target_duration - tolerance
    target_max_duration = target_duration + tolerance
    
    constraints = [
        cp.sum(weights_var) == 1,
        weights_var >= 0,
        weights_var <= 0.15,
        portfolio_duration >= target_min_duration,
        portfolio_duration <= target_max_duration
    ]
    
    # Definir o problema de otimizacao
    objective = cp.Minimize(portfolio_risk + lambda_reg * cp.sum_squares(weights_var - (1 / len(selected_assets))))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    
    # Verificar se a otimizacao foi bem-sucedida
    if prob.status not in ["infeasible", "unbounded"]:
        optimized_weights = weights_var.value
        
        # Calcular metricas do portfolio otimizado
        portfolio_return_opt = np.dot(optimized_weights, selected_assets['Taxa Indicativa'])
        portfolio_risk_opt = np.sqrt(np.dot(optimized_weights.T, np.dot(np.diag(selected_assets['Risco Medio'].values**2), optimized_weights)))
        portfolio_duration_opt = np.dot(optimized_weights, selected_assets['Duration'])
        
        resultados.append({
            'Carteira': i,
            'Taxa Inicial': portfolio_return_initial,
            'Risco Inicial': portfolio_risk_initial,
            'Duration Inicial': portfolio_duration_initial,
            'Taxa Otimizada': portfolio_return_opt,
            'Risco Otimizado': portfolio_risk_opt,
            'Duration Otimizada': portfolio_duration_opt
        })

# Converter os resultados para um DataFrame
df_resultados = pd.DataFrame(resultados)

# Exibir estatisticas gerais
print("\nEstatisticas Gerais:")
print(df_resultados.describe().round(4))

plt.figure(figsize=(14, 8))

# Plotar os pontos das carteiras nao otimizadas
plt.scatter(df_resultados['Risco Inicial'], df_resultados['Taxa Inicial'] * 100, alpha=0.5, label='Nao Otimizado', color='red')

# Plotar os pontos das carteiras otimizadas
plt.scatter(df_resultados['Risco Otimizado'], df_resultados['Taxa Otimizada'] * 100, alpha=0.5, label='Otimizado', color='green')

# Adicionar as etiquetas para cada ponto, com um deslocamento acima da bolinha
for i in range(len(df_resultados)):
    # Ponto nao otimizado
    plt.text(df_resultados['Risco Inicial'][i], df_resultados['Taxa Inicial'][i] * 100 + 0.05, f'C{i+1}', color='red', fontsize=8, ha='center')
    
    # Ponto otimizado
    plt.text(df_resultados['Risco Otimizado'][i], df_resultados['Taxa Otimizada'][i] * 100 + 0.05, f'C{i+1}', color='green', fontsize=8, ha='center')

# Configuracoes do grafico
plt.xlabel('Risco')
plt.ylabel('Taxa Esperada (%)')
plt.title('Comparacao entre Carteiras Nao Otimizadas e Otimizadas')
plt.legend()
plt.grid(True)
plt.show()


# Contar quantas carteiras nao foram otimizadas
carteiras_nao_otimizadas = 10000 - len(df_resultados)

# Comparacao de metricas medias
media_inicial = df_resultados[['Taxa Inicial', 'Risco Inicial', 'Duration Inicial']].mean()
media_otimizada = df_resultados[['Taxa Otimizada', 'Risco Otimizado', 'Duration Otimizada']].mean()

df_comparacao = pd.DataFrame({
    'Metrica': ['Taxa Esperada (%)', 'Risco Medio (%)', 'Duration (anos)', 'Carteiras Nao Otimizadas'],
    'Inicial': [media_inicial['Taxa Inicial'] * 100, media_inicial['Risco Inicial'] * 100, media_inicial['Duration Inicial'], '-'],
    'Otimizada': [media_otimizada['Taxa Otimizada'] * 100, media_otimizada['Risco Otimizado'] * 100, media_otimizada['Duration Otimizada'], carteiras_nao_otimizadas]
}).round(2)

# Exibir a tabela
print("\nTabela Comparativa de Metricas Medias:")
print(df_comparacao)

# Ajustar o grafico para nao incluir a metrica de carteiras nao otimizadas
df_comparacao_plot = df_comparacao[df_comparacao['Metrica'] != 'Carteiras Nao Otimizadas']
df_comparacao_plot.set_index('Metrica').plot(kind='bar', figsize=(10, 6), rot=0)
plt.title('Comparacao de Metricas Medias entre Carteiras Nao Otimizadas e Otimizadas')
plt.ylabel('Valores (%)')
plt.grid(axis='y')
plt.show()
