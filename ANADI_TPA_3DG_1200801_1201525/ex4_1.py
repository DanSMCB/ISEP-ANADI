# 4.1.1 Emissoes Totais de CO2 de Portugal (1900-2021)
import matplotlib.pyplot as plt
import pandas as pd

co2_data = pd.read_csv(r'CO_data.csv')
portugal_co2 = co2_data[co2_data['country'] == 'Portugal']

plt.figure(figsize=(14, 7))
plt.plot(portugal_co2['year'], portugal_co2['co2'], marker='o', linestyle='-', color='blue')
plt.title('Emissoes Totais de CO2 de Portugal (1900-2021)')
plt.xlabel('Ano')
plt.ylabel('Emissoes de CO2 (em milhoes de toneladas)')
plt.grid(True)
plt.savefig('emissoes_totais_co2_portugal_1900_2021.png')
max_co2_year = portugal_co2.loc[portugal_co2['co2'].idxmax(), 'year']
print(f'O ano de valor m�ximo de CO2 �: {max_co2_year}')
#plt.close()

# 4.1.2 Comparacao das Emissoes de CO2 de Portugal por Fonte (1900-2021)
emissions_sources = ['cement_co2', 'coal_co2', 'flaring_co2', 'gas_co2', 'methane', 'nitrous_oxide', 'oil_co2']
plt.figure(figsize=(14, 7))
for source in emissions_sources:
    plt.plot(portugal_co2['year'], portugal_co2[source], label=source)
    
plt.title('Comparacao das Emissoes de CO2 de Portugal por Fonte (1900-2021)')
plt.xlabel('Ano')
plt.ylabel('Emissoes de CO2 (em milhões de toneladas)')
plt.legend()
plt.grid(True)
plt.savefig('comparativo_emissoes_co2_portugal_por_fonte_1900_2021.png')
#plt.close()

# 4.1.3 Emissoes de CO2 per Capita: Portugal vs. Espanha (1900-2021)
spain_co2 = co2_data[co2_data['country'] == 'Spain']
portugal_co2 = portugal_co2.copy()
portugal_co2['co2_per_capita'] = portugal_co2['co2'] / portugal_co2['population'] * 1e6

spain_co2 = spain_co2.copy()
spain_co2['co2_per_capita'] = spain_co2['co2'] / spain_co2['population'] * 1e6

plt.figure(figsize=(14, 7))
plt.plot(portugal_co2['year'], portugal_co2['co2_per_capita'], label='Portugal')
plt.plot(spain_co2['year'], spain_co2['co2_per_capita'], label='Espanha')
plt.title('Emissoes de CO2 per Capita: Portugal vs. Espanha (1900-2021)')
plt.xlabel('Ano')
plt.ylabel('Emissoes de CO2 per Capita (toneladas por pessoa)')
plt.legend()
plt.grid(True)
plt.savefig('emissoes_co2_per_capita_portugal_vs_espanha_1900_2021.png')
#plt.close()

# regions of interest
regions = ['United States', 'China', 'India', 'European Union (27)', 'Russia']

# Filter data for the period 2000-2021 and the specified regions
filtered_data = co2_data[(co2_data['year'].between(2000, 2021)) & (co2_data['country'].isin(regions))]

# 4.1.4 Fazer plotting emissões CO2 de carvão
plt.figure(figsize=(12, 8))
for region in regions:
    region_data = filtered_data[filtered_data['country'] == region]
    plt.plot(region_data['year'], region_data['coal_co2'], label=region, marker='o')

plt.title('CO2 Emissions from Coal (2000-2021)')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (Mt)')
plt.legend()
plt.grid(True)
plt.savefig('plotCO2EmissionsCoal.png')
#plt.close()

# 4.1.5 Calcular médias de emissões de CO2 por país	
emissions_sources = ['cement_co2', 'coal_co2', 'flaring_co2', 'gas_co2', 'methane', 'nitrous_oxide', 'oil_co2']
average_emissions = filtered_data.groupby('country')[emissions_sources].mean().round(3)

# Plotting the table
plt.figure(figsize=(10, 6))
table = plt.table(cellText=average_emissions.values,
                  colLabels=average_emissions.columns,
                  rowLabels=average_emissions.index,
                  loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.axis('off')
plt.savefig('average_emissions_table.png')
#plt.close()