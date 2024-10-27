import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro

#4.3.1
co2_data = pd.read_csv(r'CO_data.csv')

# Filtra os dados para o período e regiões de interesse
regions_of_interest = ['Africa', 'Asia', 'South America', 'North America', 'Europe', 'Oceania']
filtered_data = co2_data[(co2_data['year'].between(2000, 2021)) & (co2_data['country'].isin(regions_of_interest))]

# Calcula as emissões de CO2 provenientes do carvão para cada região
co2_from_coal = filtered_data.pivot_table(values='coal_co2', index='year', columns='country', aggfunc='sum')

# Calcula a tabela de correlação
correlation_table = co2_from_coal.corr()

correlation_table

# Visualiza a tabela de correlação com Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_table, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlação entre Emissões de CO2 provenientes do carvão entre Regiões (2000-2021)')
plt.savefig('correlation_table_co2_from_coal.png')
plt.show()

##4.3.2
# a) Encontre o modelo de regressão linear.
years_pares = [ano for ano in range(2000, 2021, 2)]
filtered_data = co2_data[(co2_data['year'].isin(years_pares)) & (co2_data['country'].isin(['Germany', 'Russia', 'France', 'Portugal', 'Europe']))]
pivoted_data = filtered_data.pivot_table(values='coal_co2', index='year', columns='country', aggfunc='sum').dropna()
X = pivoted_data[['Germany', 'Russia', 'France', 'Portugal']]  # Variáveis independentes
y = pivoted_data['Europe']  # Variável dependente
X = sm.add_constant(X)  # Adiciona constante ao modelo
model = sm.OLS(y, X).fit()

# b) Verifica as condições sobre os resíduos.
residuos = model.resid
shapiro_test = shapiro(residuos)

# c) Verifica se existe colinearidade (VIF).
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# d) Visualização e save das figuras
# Valores Reais vs Preditos
plt.figure(figsize=(10, 6))
plt.scatter(y, model.predict(X), color='blue', label='Valores Preditos')
plt.plot(y, y, color='red', label='Identidade')
plt.title('Emissões de CO2 na Europa: Valores Reais vs Preditos')
plt.xlabel('Emissões Reais de CO2 (Europa)')
plt.ylabel('Emissões Preditas de CO2 (Europa)')
plt.legend()
plt.savefig('valores_reais_vs_preditos_corrigido.png')
plt.show()

# QQ Plot dos Resíduos
fig = sm.qqplot(residuos, line='s')
plt.title('QQ Plot dos Resíduos')
plt.savefig('qq_plot_residuos_corrigido.png')
plt.show()

# Matriz de Correlação entre Variáveis Independentes
correlation_matrix = X.iloc[:, 1:].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlação entre Variáveis Independentes')
plt.savefig('matriz_correlacao_corrigido.png')
plt.show()

# e) Estimativa para 2015 
presente_2015 = co2_data[co2_data['year'] == 2015]
presente_2015_regioes = presente_2015[presente_2015['country'].isin(['Germany', 'Russia', 'France', 'Portugal', 'Europe'])]
dados_2015 = presente_2015_regioes.set_index('country')['coal_co2']

dados_2015_ajustados = {'const': 1, 'Germany': dados_2015.loc['Germany'], 'Russia': dados_2015.loc['Russia'], 'France': dados_2015.loc['France'], 'Portugal': dados_2015.loc['Portugal']}
X_2015_ajustado = pd.DataFrame(dados_2015_ajustados, index=[0])
estimativa_2015_ajustada = model.predict(X_2015_ajustado)

valor_real_2015 = dados_2015.loc['Europe']

plt.figure(figsize=(8, 5))
plt.bar(['Estimativa 2015', 'Real 2015'], [estimativa_2015_ajustada.iloc[0], valor_real_2015], color=['blue', 'green'])
plt.title('Estimativa vs Valor Real de Emissões de CO2 em 2015')
plt.ylabel('Emissões de CO2 (toneladas)')
plt.savefig('estimativa_vs_real_2015_corrigido_final.png')
plt.show()