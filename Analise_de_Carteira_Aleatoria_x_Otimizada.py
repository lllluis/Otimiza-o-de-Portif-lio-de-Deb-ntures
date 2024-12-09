import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import random

# Definir o caminho do arquivo Excel
caminho_arquivo = r"D:\TCC\Base_Limpa.xlsx"

# Ler as abas 'IPCA' e 'CDI' em DataFrames separados
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

# Combinar os DataFrames em um único DataFrame (se necessário)
df_portfolio = df_ipca.copy()  # Apenas IPCA está sendo usado no exemplo

# Definir o mapeamento de risco médio baseado no Rating
risk_mapping = {
    'A1': 0.03,
    'A2': 0.05,
    'A3': 0.07,
    'B3': 0.10,
    'C': 0.15,
    'F': 0.20
}

# Adicionar a coluna 'Risco Médio' ao DataFrame
df_portfolio['Risco Médio'] = df_portfolio['Rating'].map(risk_mapping)

# Verificar se há algum Rating que não foi mapeado
if df_portfolio['Risco Médio'].isnull().any():
    print("Existem Ratings não mapeados. Verifique o mapeamento de risco.")
    df_portfolio = df_portfolio.dropna(subset=['Risco Médio'])

# Verificar se a Duration está em anos; se não, converter
# Para este exemplo, vou assumir que está em anos
# Se estiver em dias, descomente a linha abaixo
# df_portfolio['Duration'] = df_portfolio['Duration'] / 365

# Selecionar 20 ativos aleatoriamente
random.seed(42)  # Para reprodutibilidade
selected_assets = df_portfolio.sample(n=20, random_state=42).reset_index(drop=True)

print("Ativos Selecionados:")
print(selected_assets)

# Função para gerar alocações aleatórias com peso máximo
def generate_random_weights(num_assets, max_weight=0.15):
    weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
    weights = np.minimum(weights, max_weight)
    weights /= weights.sum()
    return weights

# Gerar alocação inicial
initial_weights = generate_random_weights(len(selected_assets), max_weight=0.15)

# Adicionar as alocações iniciais ao DataFrame
selected_assets['Alocacao Inicial (%)'] = (initial_weights * 100).round(2)

# Calcular o retorno esperado, risco médio e duration do portfólio inicial
portfolio_return_initial = np.dot(initial_weights, selected_assets['Taxa Indicativa'])
# Considerando risco como desvio padrão ponderado
portfolio_risk_initial = np.sqrt(np.dot(initial_weights.T, np.dot(np.diag(selected_assets['Risco Médio'].values**2), initial_weights)))
portfolio_duration_initial = np.dot(initial_weights, selected_assets['Duration'])

print("\nAlocação Inicial:")
print(selected_assets[['Ticker', 'Alocacao Inicial (%)']])

print(f"\nTaxa Esperada do Portfólio Inicial: {portfolio_return_initial*100:.2f}%")
print(f"Risco Médio do Portfólio Inicial: {portfolio_risk_initial*100:.2f}%")
print(f"Duration do Portfólio Inicial: {portfolio_duration_initial:.2f} anos")

# Definir variáveis para otimização
weights_var = cp.Variable(len(selected_assets))

# Objetivo: Minimizar o risco do portfólio
portfolio_risk = selected_assets['Risco Médio'].values @ weights_var
portfolio_return = selected_assets['Taxa Indicativa'].values @ weights_var
portfolio_duration = selected_assets['Duration'].values @ weights_var

# Definir o target de duration
target_duration = 1.2  # Exemplo: 3 anos

# Definir uma tolerância para o target de duration
tolerance = 0.5  # Permite variação de ±0.5 anos

# Definir os inputs para duration target
target_min_duration = target_duration - tolerance  # 2.5 anos
target_max_duration = target_duration + tolerance  # 3.5 anos

# Parâmetro para regularização (ajuste conforme necessário)

lambda_reg = 200  # Ajustado para evitar inviabilidade

# Definir o objetivo de otimização (minimizar risco + regularização)
# Objetivo: Minimizar o risco e penalizar desvios da alocação uniforme
objective = cp.Minimize(portfolio_risk + lambda_reg * cp.sum_squares(weights_var - (1 / len(selected_assets))))

# Restrições:
constraints = [
    cp.sum(weights_var) == 1,                # Soma das alocações deve ser 1
    weights_var >= 0,                         # Nenhum peso negativo
    weights_var <= 0.15,                      # Peso máximo de 15% por ativo
    portfolio_duration >= target_min_duration,  # Duration mínima
    portfolio_duration <= target_max_duration   # Duration máxima
]

# Definir e resolver o problema
prob = cp.Problem(objective, constraints)

# Resolver com solver alternativo e verbose para diagnóstico
prob.solve(solver=cp.SCS, verbose=True)  # Tente SCS se OSQP falhar

# Verificar se a otimização foi bem-sucedida
if prob.status not in ["infeasible", "unbounded"]:
    # Obter os pesos otimizados
    optimized_weights = weights_var.value

    # Adicionar as alocações otimizadas ao DataFrame
    selected_assets['Alocacao Otimizada (%)'] = (optimized_weights * 100).round(2)

    # Calcular o retorno esperado, risco médio e duration do portfólio otimizado
    portfolio_return_opt = np.dot(optimized_weights, selected_assets['Taxa Indicativa'])
    portfolio_risk_opt = np.sqrt(np.dot(optimized_weights.T, np.dot(np.diag(selected_assets['Risco Médio'].values**2), optimized_weights)))
    portfolio_duration_opt = np.dot(optimized_weights, selected_assets['Duration'])

    print("\nAlocação Otimizada:")
    print(selected_assets[['Ticker', 'Alocacao Otimizada (%)']])

    print(f"\nTaxa Esperada do Portfólio Otimizado: {portfolio_return_opt*100:.2f}%")
    print(f"Risco Médio do Portfólio Otimizado: {portfolio_risk_opt*100:.2f}%")
    print(f"Duration do Portfólio Otimizado: {portfolio_duration_opt:.2f} anos")

    # Tabela comparativa das alocações
    comparison = selected_assets[['Ticker', 'Alocacao Inicial (%)', 'Alocacao Otimizada (%)']]
    print("\nTabela Comparativa das Alocações:")
    print(comparison.to_string(index=False))

    # Tabela comparativa de métricas
    metrics_comparison = pd.DataFrame({
        'Métrica': ['Taxa Esperada (%)', 'Risco Médio (%)', 'Duration (anos)'],
        'Inicial': [portfolio_return_initial * 100, portfolio_risk_initial * 100, portfolio_duration_initial],
        'Otimizada': [portfolio_return_opt * 100, portfolio_risk_opt * 100, portfolio_duration_opt]
    }).round(2)
    print("\nTabela Comparativa de Taxa, Risco e Duration:")
    print(metrics_comparison.to_string(index=False))

    # Plotar a comparação das alocações
    comparison.plot(x='Ticker', kind='bar', figsize=(14,7))
    plt.title('Comparação das Alocações Iniciais e Otimizadas')
    plt.ylabel('Alocação (%)')
    plt.xticks(rotation=45)
    plt.legend(['Alocação Inicial', 'Alocacao Otimizada'])
    plt.tight_layout()
    plt.show()

    # --- Plotar a Fronteira Eficiente ---
    # Definir uma gama de retornos alvo
    target_returns = np.linspace(selected_assets['Taxa Indicativa'].min(), selected_assets['Taxa Indicativa'].max(), 50)
    efficient_risks = []
    efficient_returns = []

    for r in target_returns:
        # Definir o objetivo para a fronteira eficiente
        objective_eff = cp.Minimize(portfolio_risk + lambda_reg * cp.sum_squares(weights_var - (1 / len(selected_assets))))

        # Restrições para cada ponto na fronteira
        constraints_eff = [
            cp.sum(weights_var) == 1,
            weights_var >= 0,
            weights_var <= 0.15,
            portfolio_return == r,
            portfolio_duration >= target_min_duration,
            portfolio_duration <= target_max_duration
        ]

        # Definir e resolver o problema
        prob_eff = cp.Problem(objective_eff, constraints_eff)
        prob_eff.solve(solver=cp.SCS, verbose=False)

        if prob_eff.status not in ["infeasible", "unbounded"]:
            # Calcular o risco para este portfólio eficiente
            risk_eff = np.sqrt(np.dot(weights_var.value.T, np.dot(np.diag(selected_assets['Risco Médio'].values**2), weights_var.value)))
            efficient_risks.append(risk_eff)
            efficient_returns.append(r * 100)  # Converter para porcentagem

    # Plotar a fronteira eficiente
    plt.figure(figsize=(10, 6))
    plt.plot(efficient_risks, efficient_returns, label='Fronteira Eficiente', color='blue')
    plt.scatter(portfolio_risk_initial, portfolio_return_initial * 100, color='red', label='Carteira Inicial', s=100, edgecolors='black')
    plt.scatter(portfolio_risk_opt, portfolio_return_opt * 100, color='green', label='Carteira Otimizada', s=100, edgecolors='black')
    plt.xlabel('Risco (Desvio Padrão)')
    plt.ylabel('Taxa Esperada (%)')
    plt.title('Fronteira Eficiente de Markowitz e Carteiras')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Problema de otimização é inviável. Verifique as restrições e os dados.")
