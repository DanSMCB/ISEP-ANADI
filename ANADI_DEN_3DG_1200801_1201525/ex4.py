import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from scipy.stats import ttest_rel
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV

# Alínea 1: Carregar o dataset e realizar análise preliminar
file_path = 'Dados_Trabalho_TP2.csv'
data = pd.read_csv(file_path)

# Remover a coluna "Unnamed: 0" se existir
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

# Verificar as dimensões do dataset
data_dimensions = data.shape
print("Dimensões do Dataset:", data_dimensions)

# Obter um sumário dos dados
data_summary = data.describe(include='all')
print("Sumário dos Dados:")
print(data_summary)

# Exibir as primeiras linhas do dataset para entender a estrutura dos dados
print("Primeiras linhas do Dataset:")
print(data.head())

# Alínea 2: Derivar o atributo "IMC"
data['IMC'] = data['Peso'] / (data['Altura'] ** 2)
print("Primeiras linhas com IMC calculado:")
print(data[['Peso', 'Altura', 'IMC']].head())

# Alínea 3: Analisar os atributos do conjunto de dados mais significativos
# Converter apenas as colunas categóricas necessárias para numéricas
categorical_cols = ['Genero', 'Historico_obesidade_familiar', 'FCCAC', 'CCER', 'Fumador', 'MCC', 'CBA', 'TRANS', 'Label']
data[categorical_cols] = data[categorical_cols].apply(lambda x: x.astype('category').cat.codes)

# Histogramas das variáveis numéricas
data.hist(figsize=(15, 10))
plt.suptitle('Distribuição das Variáveis Numéricas')
plt.show()

# Gráficos de dispersão entre Idade, Peso, Altura e IMC
sns.pairplot(data[['Idade', 'Peso', 'Altura', 'IMC']])
plt.suptitle('Gráficos de Dispersão entre Variáveis')
plt.show()

# Boxplot para visualizar a distribuição do IMC em relação a variáveis categóricas
plt.figure(figsize=(12, 8))
sns.boxplot(x='Genero', y='IMC', data=data)
plt.title('Boxplot do IMC em relação ao Género')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='Historico_obesidade_familiar', y='IMC', data=data)
plt.title('Boxplot do IMC em relação ao Histórico de Obesidade Familiar')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='Fumador', y='IMC', data=data)
plt.title('Boxplot do IMC em relação ao Fumar')
plt.show()

# Alínea 4: Limpeza de dados e normalização
# Remover entradas com valores ausentes
data = data.dropna()

# Normalização de atributos numéricos
scaler = StandardScaler()
numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_features] = scaler.fit_transform(data[numeric_features])

print("Primeiras linhas após normalização de todos os atributos numéricos:")
print(data.head())

# Alínea 5: Criar um diagrama de correlação entre todos os atributos
correlation_matrix = data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# Alínea 6: Obter um modelo de regressão linear simples para a variável “IMC”
# Preparar dados para o modelo de regressão
X = data[['Idade']]  # Usar apenas colunas numéricas
y = data['IMC']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Criar e ajustar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Coeficiente e intercepto da função linear
coef = model.coef_[0]
intercept = model.intercept_

print(f"Função Linear: IMC = {coef:.4f} * Idade + {intercept:.4f}")

# Fazer previsões com o modelo ajustado
y_pred = model.predict(X_test)

# Visualizar a reta de regressão e o diagrama de dispersão
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['Idade'], y=y_test, label='Dados Reais')
sns.lineplot(x=X_test['Idade'], y=y_pred, color='red', label='Modelo de Regressão Linear')
plt.title('Regressão Linear Simples - Idade vs IMC')
plt.xlabel('Idade')
plt.ylabel('IMC')
plt.legend()
plt.show()

# Calcular MAE e RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Lista de variáveis preditoras disponíveis no dataset
predictors = ['Altura', 'Peso', 'FCCAC', 'FCV', 'NRP', 'CCER', 'CA', 'MCC', 'FAF', 'TUDE', 'CBA', 'TRANS']

# Dicionário para armazenar os resultados dos modelos
results = {}

for predictor in predictors:
    X = data[[predictor]]
    y = data['IMC']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    results[predictor] = {'MAE': mae, 'RMSE': rmse, 'Coeficiente': model.coef_[0], 'Intercepto': model.intercept_}

# Exibir os resultados
for predictor, result in results.items():
    print(f"Variável: {predictor}")
    print(f"  Coeficiente: {result['Coeficiente']:.4f}")
    print(f"  Intercepto: {result['Intercepto']:.4f}")
    print(f"  MAE: {result['MAE']:.4f}")
    print(f"  RMSE: {result['RMSE']:.4f}")
    print("")

# Alínea 7a: Regressão Linear Múltipla
# Preparar dados para regressão linear múltipla
X = data.drop(columns=['IMC'])  # Utilizar todos os atributos como preditores
y = data['IMC']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Criar e ajustar o modelo de regressão linear múltipla
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Fazer previsões
y_pred_linear = linear_model.predict(X_test)

# Avaliar o modelo
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = root_mean_squared_error(y_test, y_pred_linear)

print(f"Regressão Linear Múltipla - MAE: {mae_linear:.4f}, RMSE: {rmse_linear:.4f}")

# Alínea 7b: Árvore de Regressão
# Criar e ajustar a árvore de regressão com parâmetros padrão
tree_model_default = DecisionTreeRegressor(random_state=42)
tree_model_default.fit(X_train, y_train)

# Fazer previsões
y_pred_tree_default = tree_model_default.predict(X_test)

# Avaliar o modelo com parâmetros padrão
mae_tree_default = mean_absolute_error(y_test, y_pred_tree_default)
rmse_tree_default = root_mean_squared_error(y_test, y_pred_tree_default)

print(f"Árvore de Regressão (Padrão) - MAE: {mae_tree_default:.4f}, RMSE: {rmse_tree_default:.4f}")

# Visualizar a árvore de regressão com parâmetros padrão
plt.figure(figsize=(20, 10))
plot_tree(tree_model_default, filled=True, feature_names=X.columns.tolist(), rounded=True)
plt.title('Árvore de Regressão (Padrão)')
plt.show()

# Definir o grid de hiperparâmetros
param_grid = {
    'max_depth': [3, 5],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2 , 4]
}

# Configurar o GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42), param_grid=param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Encontrar os melhores parâmetros
best_params = grid_search.best_params_
print(f"Melhores parâmetros: {best_params}")

# Treinar o melhor modelo
best_tree_model = grid_search.best_estimator_
best_tree_model.fit(X_train, y_train)

# Fazer previsões
y_pred_tree_optimized = best_tree_model.predict(X_test)

# Avaliar o modelo otimizado
mae_tree_optimized = mean_absolute_error(y_test, y_pred_tree_optimized)
rmse_tree_optimized = root_mean_squared_error(y_test, y_pred_tree_optimized)

print(f"Árvore de Regressão (Otimizada) - MAE: {mae_tree_optimized:.4f}, RMSE: {rmse_tree_optimized:.4f}")

# Visualizar a árvore de regressão otimizada
plt.figure(figsize=(20, 10))
plot_tree(best_tree_model, filled=True, feature_names=X.columns.tolist(), rounded=True)
plt.title('Árvore de Regressão (Otimizada)')
plt.show()


# Alínea 7c: Rede Neural MLPRegressor
# Função para criar e treinar a rede neural
def create_and_train_model(X_train, y_train, X_test, y_test, hidden_layers):
    model = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    return model, mae, rmse, y_pred

# Avaliar diferentes arquiteturas de rede
configurations = [(1,), (3,), (6, 3), (100, 50)]
results = []

for config in configurations:
    model, mae, rmse, y_pred = create_and_train_model(X_train, y_train, X_test, y_test, config)
    results.append((config, mae, rmse))
    print(f"Configuração {config} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Melhor configuração
best_config = min(results, key=lambda x: x[1])
print(f"Melhor Configuração: {best_config[0]}")

# Visualizar a melhor rede obtida
best_model_mlp, _, _, _ = create_and_train_model(X_train, y_train, X_test, y_test, best_config[0])

# Plotar a árvore da rede neural treinada
plt.figure(figsize=(10, 6))
plt.title(f'MLPRegressor com configuração {best_config[0]}')
plt.plot(best_model_mlp.loss_curve_)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Alínea 8: Comparação dos resultados
# Função para calcular e exibir MAE e RMSE
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        results[name] = {'MAE': mae, 'RMSE': rmse}
        print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return results

# Avaliar os modelos
models = {
    'Regressão Linear Múltipla': linear_model,
    'Árvore de Regressão (Padrão)': tree_model_default,
    'Árvore de Regressão (Otimizada)': best_tree_model,
    'Rede Neuronal MLPRegressor': best_model_mlp,
}

results = evaluate_models(models, X_test, y_test)

# Exibir os resultados
results_df = pd.DataFrame(results).T
print(results_df)

# Alínea 9: Justificação de resultados significativos
# Assumindo que os dois melhores modelos são a Regressão Linear Múltipla e a Rede Neuronal MLP
# Verificar os p-valores para MAE e RMSE

_, p_value_mae = ttest_rel(y_test, y_pred_linear)
_, p_value_rmse = ttest_rel(y_test, best_model_mlp.predict(X_test))

print(f"P-valor MAE (Regressão Linear Múltipla vs Rede Neuronal MLP): {p_value_mae:.4f}")
print(f"P-valor RMSE (Regressão Linear Múltipla vs Rede Neuronal MLP): {p_value_rmse:.4f}")

# Conclusão sobre significância estatística
alpha = 0.05
if p_value_mae < alpha:
    print("A diferença entre os modelos é estatisticamente significativa para MAE.")
else:
    print("A diferença entre os modelos não é estatisticamente significativa para MAE.")

if p_value_rmse < alpha:
    print("A diferença entre os modelos é estatisticamente significativa para RMSE.")
else:
    print("A diferença entre os modelos não é estatisticamente significativa para RMSE.")
