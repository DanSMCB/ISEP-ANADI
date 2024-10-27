import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway
import matplotlib.pyplot as plt

#4.2.1
# Carregar os dados
co2_data = pd.read_csv(r'CO_data.csv')

# Define a semente e seleciona anos aleatórios para a amostra
np.random.seed(100)
years = pd.Series(np.arange(1900, 2022))
sampleyears1 = years.sample(n=30, replace=False)

# Filtra os dados para Portugal e Hungria com base nos anos da amostra
portugal_gdp_sample = co2_data[(co2_data['country'] == 'Portugal') & (co2_data['year'].isin(sampleyears1))]['gdp']
hungary_gdp_sample = co2_data[(co2_data['country'] == 'Hungary') & (co2_data['year'].isin(sampleyears1))]['gdp']

# Realiza o teste t de duas amostras para médias independentes
t_stat, p_value = stats.ttest_ind(portugal_gdp_sample.dropna(), hungary_gdp_sample.dropna(), alternative='greater')

t_stat, p_value

plt.figure(figsize=(10, 6))
plt.boxplot([portugal_gdp_sample.dropna(), hungary_gdp_sample.dropna()], labels=['Portugal', 'Hungria'])
plt.title('Comparação do PIB entre Portugal e Hungria')
plt.ylabel('PIB')
plt.grid(True)
plt.savefig('teste_t_PIB_Portugal_Hungria.png')
plt.show()


''' 
Para a seção 4.2.1, realizamos um teste t de duas amostras para médias independentes, 
comparando o PIB de Portugal e da Hungria com base nos anos da amostra selecionada (sampleyears1). 
O valor de t estatístico obtido é aproximadamente 0.181 e o valor-p é aproximadamente 0.428.
Considerando um nível de significância de 5% (α = 0.05), o valor-p é maior que 0.05,
o que indica que não podemos rejeitar a hipótese nula. 
Isso sugere que não há evidências suficientes para afirmar que a média do PIB de Portugal 
foi significativamente superior à média do PIB da Hungria no período de 1900-2021,
 com base nos anos selecionados pela amostra sampleyears1.
'''

#4.2.2
# Define a semente e seleciona anos aleatórios para as duas novas amostras
np.random.seed(55)
sampleyears2 = years.sample(n=12, replace=False)

np.random.seed(85)
sampleyears3 = years.sample(n=12, replace=False)

# Filtra os dados para Portugal e Hungria com base nos anos das amostras
portugal_gdp_sample2 = co2_data[(co2_data['country'] == 'Portugal') & (co2_data['year'].isin(sampleyears2))]['gdp']
hungary_gdp_sample3 = co2_data[(co2_data['country'] == 'Hungary') & (co2_data['year'].isin(sampleyears3))]['gdp']

# Realiza o teste t de duas amostras para médias independentes
t_stat2, p_value2 = stats.ttest_ind(portugal_gdp_sample2.dropna(), hungary_gdp_sample3.dropna(), alternative='greater')

t_stat2, p_value2

plt.figure(figsize=(10, 6))
means = [portugal_gdp_sample2.mean(), hungary_gdp_sample3.mean()]
plt.bar(['Portugal sampleyears2', 'Hungria sampleyears3'], means, color=['blue', 'red'])
plt.title('Média do PIB de Portugal e Hungria com Amostras Diferentes')
plt.ylabel('Média do PIB')
plt.grid(axis='y')
plt.savefig('teste_t_PIB_Portugal_Hungria_amostras_diferente.png')
plt.show()


'''
Para a seção 4.2.2, realizamos um teste t de duas amostras para médias independentes, 
utilizando duas amostras distintas para Portugal (sampleyears2) e Hungria (sampleyears3).
O valor de t estatístico obtido é aproximadamente 0.242 e o valor-p é aproximadamente 0.406.
Novamente, considerando um nível de significância de 5% (α = 0.05), o valor-p é maior que 0.05,
indicando que não podemos rejeitar a hipótese nula. 
Portanto, não há evidências suficientes para afirmar que a média do PIB de Portugal 
foi significativamente superior à média do PIB da Hungria no período de 1900-2021, 
com base nas amostras selecionadas sampleyears2 para Portugal e sampleyears3 para Hungria.
'''

#4.2.3
# Filtra os dados para os países e anos de interesse
regions = ['United States', 'Russia', 'China', 'India', 'European Union (27)']
filtered_co2 = co2_data[(co2_data['country'].isin(regions)) & (co2_data['year'].isin(sampleyears2))]

# Prepara os dados para a ANOVA
us_co2 = filtered_co2[filtered_co2['country'] == 'United States']['co2'].dropna()
russia_co2 = filtered_co2[filtered_co2['country'] == 'Russia']['co2'].dropna()
china_co2 = filtered_co2[filtered_co2['country'] == 'China']['co2'].dropna()
india_co2 = filtered_co2[filtered_co2['country'] == 'India']['co2'].dropna()
eu_co2 = filtered_co2[filtered_co2['country'] == 'European Union (27)']['co2'].dropna()

# Realiza ANOVA
anova_result = f_oneway(us_co2, russia_co2, china_co2, india_co2, eu_co2)

anova_result

plt.figure(figsize=(12, 6))
regions_means = [us_co2.mean(), russia_co2.mean(), china_co2.mean(), india_co2.mean(), eu_co2.mean()]
plt.bar(['Estados Unidos', 'Rússia', 'China', 'Índia', 'European Union'], regions_means, color=['blue', 'green', 'red', 'orange', 'purple'])
plt.title('Médias de Emissões de CO2 por Região')
plt.ylabel('Emissões de CO2')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig('anova.png')
plt.show()

'''
Análise Visual da ANOVA das Emissões de CO2:

O gráfico de barras apresenta as médias de emissões de CO2 para cinco regiões-chave: Estados Unidos, Rússia, China, Índia e União Europeia, baseadas nos anos amostrados. Esta visualização é crucial para entender rapidamente as diferenças entre as emissões médias de CO2 de regiões significativamente impactantes no cenário global de mudanças climáticas.

Observando as alturas das barras, podemos identificar quais regiões possuem, em média, maiores emissões de CO2. Uma barra mais alta indica maior emissão média de CO2, o que pode sinalizar maior impacto ambiental em termos de contribuição para o aquecimento global.

A partir deste gráfico, podemos facilmente comparar as regiões entre si, destacando aquelas que, durante os anos selecionados, contribuíram mais significativamente para as emissões de CO2. Este tipo de análise é fundamental para políticas ambientais, discussões sobre mudanças climáticas e para entender o papel de diferentes economias globais na emissão de gases de efeito estufa.

Além disso, a visualização suporta a interpretação dos resultados da ANOVA, fornecendo um meio intuitivo de avaliar a variação das emissões entre as regiões. Se o resultado da ANOVA indicar diferenças estatisticamente significativas, o gráfico de barras complementa essa conclusão, destacando visualmente essas diferenças. Por outro lado, se a ANOVA não encontrar diferenças significativas, o gráfico pode ajudar a investigar se isso se deve à similaridade nas médias de emissões ou a outros fatores, como variabilidade dentro das regiões.

Em resumo, este gráfico não apenas facilita a interpretação dos resultados estatísticos, mas também serve como uma ferramenta poderosa para comunicar a urgência e a necessidade de ações direcionadas para as regiões com maiores emissões médias de CO2, direcionando esforços de mitigação onde eles são mais necessários.
'''
