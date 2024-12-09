from flask import Flask, request, render_template_string, url_for
import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
import random

app = Flask(__name__)

# Função para pré-processar os dados
def preprocess_dataframe(df):
    # Garantir que as colunas relevantes estão no formato adequado
    df['Data Ref'] = pd.to_datetime(df['Data Ref'])
    df['Vencimento'] = pd.to_datetime(df['Vencimento'])
    
    # Excluir linhas onde a data de vencimento é anterior à data de hoje
    df = df[df['Vencimento'] >= pd.Timestamp.today()]
    
    # Identificar os registros mais recentes para a Taxa Indicativa, Duration e Rating com base na Data Ref
    df_processed = df.loc[df.groupby('Ticker')['Data Ref'].idxmax(), ['Ticker', 'Taxa Indicativa', 'Duration', 'Rating', 'Data Ref']]
    
    return df_processed

# Definir o caminho do arquivo Excel
# Atualize este caminho conforme a localização do seu arquivo
caminho_arquivo = r"D:\TCC\Base_Limpa.xlsx"

# Ler as abas IPCA e CDI em DataFrames separados
df_ipca = pd.read_excel(caminho_arquivo, sheet_name='IPCA')
df_cdi = pd.read_excel(caminho_arquivo, sheet_name='CDI')

# Processar os dataframes IPCA e CDI
df_ipca = preprocess_dataframe(df_ipca)
df_cdi = preprocess_dataframe(df_cdi)

# Combinar os DataFrames em um único DataFrame
df_portfolio = pd.concat([df_ipca, df_cdi], ignore_index=True)

# Definir o mapeamento de risco médio baseado no Rating
risk_mapping = {
    'A1': 0.03,
    'A2': 0.05,
    'A3': 0.07,
    'B3': 0.10,
    'C': 0.15,
    'F': 0.20
}

# Adicionar a coluna Risco Médio ao DataFrame
df_portfolio['Risco Medio'] = df_portfolio['Rating'].map(risk_mapping)

# Verificar se há algum Rating que não foi mapeado
if df_portfolio['Risco Medio'].isnull().any():
    print("Existem Ratings não mapeados. Verifique o mapeamento de risco.")
    df_portfolio = df_portfolio.dropna(subset=['Risco Medio'])

# Lista de tickers disponíveis
tickers = df_portfolio['Ticker'].unique()

# Templates HTML como strings
index_html = '''
<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <title>Simulação e Otimização de Portfólio de Debêntures</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css">
    <style>
        body {
            background-color: #f8f9fa;
        }
        .header {
            margin-top: 20px;
            margin-bottom: 40px;
            position: relative;
        }
        .header h1 {
            color: #343a40;
        }
        .logo {
            position: absolute;
            top: 0;
            right: 0;
        }
        .logo img {
            height: 80px;
        }
        .form-group label {
            font-weight: bold;
        }
        .btn-primary {
            background-color: #007bff;
            border: none;
            margin-top: 20px;
        }
        .alert {
            margin-top: 20px;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            color: #6c757d;
        }
        .theory {
            margin-top: 50px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Simulação e Otimização de Portfólio de Debêntures</h1>
            <div class="logo">
                <img src="{{ url_for('static', filename='Logotipo_da_POLI-USP.jpg') }}" alt="Logo">
            </div>
        </div>
        {% if error %}
            <div class="alert alert-danger" role="alert">
                {{ error }}
            </div>
        {% endif %}
        <form method="post">
            <div class="form-group">
                <label for="tickers">Escolha os Tickers:</label>
                <select multiple class="form-control" id="tickers" name="tickers" required>
                    {% for ticker in tickers %}
                        <option value="{{ ticker }}">{{ ticker }}</option>
                    {% endfor %}
                </select>
                <small class="form-text text-muted">Segure a tecla Ctrl (ou Command no Mac) para selecionar múltiplos códigos.</small>
            </div>
            <div class="form-group">
                <label for="target_duration">Duration Médio Desejado:</label>
                <input type="number" step="0.01" class="form-control" id="target_duration" name="target_duration" required>
            </div>
            <button type="submit" class="btn btn-primary btn-block">Calcular Portfólio</button>
        </form>
        <div class="footer">
            <p>&copy; {{ current_year }} Universidade de São Paulo</p>
        </div>
        <div class="theory">
            <h3>Modelo de Markowitz</h3>
            <p>
                O modelo de Markowitz, também conhecido como teoria moderna de portfólios, é um modelo de otimização de carteiras que busca maximizar o retorno esperado para um dado nível de risco ou minimizar o risco para um dado nível de retorno esperado. O modelo considera a diversificação de ativos para reduzir o risco total da carteira.
            </p>
            <p>
                <strong>Matematicamente:</strong>
            </p>
            <p>
                Maximize o retorno esperado do portfólio:
                $$ E(R_p) = \\sum_{i=1}^n w_i E(R_i) $$
            </p>
            <p>
                Minimize o risco (variância) do portfólio:
                $$ \\sigma_p^2 = \\sum_{i=1}^n \\sum_{j=1}^n w_i w_j \\sigma_{ij} $$
            </p>
            <p>
                Sujeito às restrições:
            </p>
            <ul>
                <li>Soma dos pesos igual a 1:
                    $$ \\sum_{i=1}^n w_i = 1 $$
                </li>
                <li>Duration médio do portfólio igual ao desejado:
                    $$ D_p = \\sum_{i=1}^n w_i D_i = D_{desejado} $$
                </li>
                <li>Pesos não negativos (sem venda a descoberto):
                    $$ w_i \\geq 0 $$
                </li>
            </ul>
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script id="MathJax-script" async
                src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
            </script>
        </div>
    </div>
</body>
</html>
'''

result_html = '''
<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <title>Resultado do Portfólio</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css">
    <style>
        body {
            background-color: #f8f9fa;
        }
        h1 {
            margin-top: 50px;
            color: #343a40;
        }
        table {
            margin-top: 30px;
        }
        .btn-secondary {
            margin-top: 20px;
        }
        th {
            background-color: #343a40;
            color: #fff;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            color: #6c757d;
        }
        .theory {
            margin-top: 50px;
        }
        .plot {
            margin-top: 30px;
            text-align: center;
        }
        .plot img {
            max-width: 100%;
            height: auto;
        }
    </style>
    <!-- MathJax -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
</head>
<body>
    <div class="container">
        <h1 class="text-center">Portfólio Sugerido</h1>
        <p class="text-center">Retorno Esperado da Carteira: {{ '{:.2f}'.format(portfolio_return * 100) }}%</p>
        <p class="text-center">Risco Médio da Carteira: {{ '{:.4f}'.format(portfolio_risk) }}</p>
        <p class="text-center">Duration Médio da Carteira: {{ '{:.2f}'.format(portfolio_duration) }}</p>
        <div class="plot">
            <h3>Gráfico de Dispersão</h3>
            <img src="data:image/png;base64,{{ plot_url }}" alt="Gráfico de Dispersão">
        </div>
        <table class="table table-striped mt-4">
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>% de Alocação</th>
                    <th>% Taxa Indicativa</th>
                    <th>Duration</th>
                    <th>Rating</th>
                </tr>
            </thead>
            <tbody>
                {% for index, row in portfolio.iterrows() %}
                <tr>
                    <td>{{ row['Ticker'] }}</td>
                    <td>{{ '{:.2%}'.format(row['Peso']) }}</td>
                    <td>{{ '{:.2%}'.format(row['Taxa Indicativa']) }}</td>
                    <td>{{ row['Duration'] }}</td>
                    <td>{{ row['Rating'] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        <div class="text-center">
            <a href="/" class="btn btn-secondary">Voltar</a>
        </div>
        <div class="footer">
            <p>&copy; {{ current_year }} Universidade de São Paulo</p>
        </div>
        <div class="theory">
            <h3>Modelo de Markowitz</h3>
            <p>
                O modelo de Markowitz, também conhecido como teoria moderna de portfólios, é um modelo de otimização de carteiras que busca maximizar o retorno esperado para um dado nível de risco ou minimizar o risco para um dado nível de retorno esperado. O modelo considera a diversificação de ativos para reduzir o risco total da carteira.
            </p>
            <p>
                <strong>Matematicamente:</strong>
            </p>
            <p>
                Maximize o retorno esperado do portfólio:
                $$ E(R_p) = \\sum_{i=1}^n w_i E(R_i) $$
            </p>
            <p>
                Minimize o risco (variância) do portfólio:
                $$ \\sigma_p^2 = \\sum_{i=1}^n \\sum_{j=1}^n w_i w_j \\sigma_{ij} $$
            </p>
            <p>
                Sujeito às restrições:
            </p>
            <ul>
                <li>Soma dos pesos igual a 1:
                    $$ \\sum_{i=1}^n w_i = 1 $$
                </li>
                <li>Duration médio do portfólio igual ao desejado:
                    $$ D_p = \\sum_{i=1}^n w_i D_i = D_{desejado} $$
                </li>
                <li>Pesos não negativos (sem venda a descoberto):
                    $$ w_i \\geq 0 $$
                </li>
            </ul>
        </div>
    </div>
</body>
</html>
'''

def generate_scatter_plot(portfolio_data, optimized_weights):
    """
    Gera um gráfico de dispersão com as carteiras iniciais e a carteira otimizada.
    Retorna a imagem codificada em base64.
    """
    num_random = 100  # Número de carteiras aleatórias para plotar
    np.random.seed(42)  # Para reprodutibilidade

    # Gerar carteiras aleatórias
    random_weights = np.random.dirichlet(np.ones(len(portfolio_data)), size=num_random)
    random_returns = random_weights @ portfolio_data['Taxa Indicativa'].values
    random_risks = random_weights @ portfolio_data['Risco Medio'].values

    # Carteira otimizada
    optimized_return = optimized_weights @ portfolio_data['Taxa Indicativa'].values
    optimized_risk = optimized_weights @ portfolio_data['Risco Medio'].values

    # Plotar
    plt.figure(figsize=(10, 6))
    plt.scatter(random_risks, random_returns * 100, alpha=0.5, label='Carteiras Aleatórias', color='blue')
    plt.scatter(optimized_risk, optimized_return * 100, color='red', label='Carteira Otimizada', s=100, marker='X')
    plt.xlabel('Risco Médio')
    plt.ylabel('Retorno Esperado (%)')
    plt.title('Comparação de Carteiras')
    plt.legend()
    plt.grid(True)

    # Salvar o plot em um buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return plot_data

@app.route('/', methods=['GET', 'POST'])
def index():
    current_year = datetime.now().year
    if request.method == 'POST':
        selected_tickers = request.form.getlist('tickers')
        target_duration = request.form.get('target_duration')  # Input retrieval

        # Validações
        if not selected_tickers:
            error = 'Por favor, selecione pelo menos um ticker.'
            return render_template_string(index_html, tickers=tickers, error=error, current_year=current_year)
        if not target_duration:
            error = 'Por favor, insira o Duration Médio Desejado.'
            return render_template_string(index_html, tickers=tickers, error=error, current_year=current_year)
        else:
            try:
                target_duration = float(target_duration)
            except ValueError:
                error = 'Duration Médio Desejado inválido.'
                return render_template_string(index_html, tickers=tickers, error=error, current_year=current_year)

        # Filtrar dados com base nos tickers selecionados
        portfolio_data = df_portfolio[df_portfolio['Ticker'].isin(selected_tickers)].copy()

        # Verificar se há dados disponíveis
        if portfolio_data.empty:
            error = 'Não há dados disponíveis para os tickers selecionados.'
            return render_template_string(index_html, tickers=tickers, error=error, current_year=current_year)

        # Parâmetros para otimização
        lambda_reg = 1.2  # Regularização, ajustável conforme necessidade
        tolerance = 0.5    # Tolerância para a duration

        # Preparar variáveis para CVXPY
        num_assets = len(portfolio_data)
        if num_assets == 0:
            error = 'Nenhum ativo selecionado para otimização.'
            return render_template_string(index_html, tickers=tickers, error=error, current_year=current_year)

        # Definir as variáveis de otimização
        weights = cp.Variable(num_assets)

        # Funções para retorno e risco
        portfolio_return = portfolio_data['Taxa Indicativa'].values @ weights
        portfolio_risk = portfolio_data['Risco Medio'].values @ weights
        portfolio_duration = portfolio_data['Duration'].values @ weights

        # Função objetivo: minimizar risco + lambda * sum_squares(weights - 1/N)
        objective = cp.Minimize(portfolio_risk + lambda_reg * cp.sum_squares(weights - (1 / num_assets)))

        # Restrições
        constraints = [
            cp.sum(weights) == 1,                            # Soma dos pesos igual a 1
            weights >= 0,                                     # Pesos não negativos
            weights <= 0.15,                                  # Peso máximo por ativo
            portfolio_duration >= (target_duration - tolerance),  # Duration mínima
            portfolio_duration <= (target_duration + tolerance)   # Duration máxima
        ]

        # Definir e resolver o problema
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        # Verificar se a otimização foi bem-sucedida
        if prob.status not in ["infeasible", "unbounded"]:
            optimized_weights = weights.value

            # Adicionar os pesos otimizados ao DataFrame
            portfolio_data = portfolio_data.copy()
            portfolio_data['Peso'] = optimized_weights

            # Calcular métricas do portfólio otimizado
            portfolio_return_value = portfolio_return.value
            portfolio_risk_value = portfolio_risk.value
            portfolio_duration_value = portfolio_duration.value

            # Gerar o gráfico de dispersão
            plot_url = generate_scatter_plot(portfolio_data, optimized_weights)

            return render_template_string(result_html, 
                                          portfolio=portfolio_data, 
                                          portfolio_return=portfolio_return_value, 
                                          portfolio_risk=portfolio_risk_value,
                                          portfolio_duration=portfolio_duration_value,
                                          plot_url=plot_url,
                                          current_year=current_year)
        else:
            error = 'Não foi possível otimizar o portfólio com os dados fornecidos e as restrições impostas.'
            return render_template_string(index_html, tickers=tickers, error=error, current_year=current_year)
    else:
        return render_template_string(index_html, tickers=tickers, error=None, current_year=current_year)

if __name__ == '__main__':
    app.run(debug=True)
