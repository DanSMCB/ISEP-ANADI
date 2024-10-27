import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import ttest_rel
from sklearn.feature_selection import SelectKBest, f_classif

# Carregar e preparar o dataset
file_path = 'Dados_Trabalho_TP2.csv'
data = pd.read_csv(file_path)
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

data['IMC'] = data['Peso'] / (data['Altura'] ** 2)
categorical_cols = ['Genero', 'Historico_obesidade_familiar', 'FCCAC', 'CCER', 'Fumador', 'MCC', 'CBA', 'TRANS', 'Label']
data[categorical_cols] = data[categorical_cols].apply(lambda x: x.astype('category').cat.codes)

# Normalização de atributos numéricos
scaler = StandardScaler()
features_to_scale = ['Idade', 'Peso', 'Altura', 'IMC']
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Seleção de atributos (para simplificar, vamos usar todos os atributos)
X = data.drop(columns=['Label'])
y = data['Label']

# Configuração do k-fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# 1.1. Árvores de Decisão
tree_clf = DecisionTreeClassifier(random_state=42)
param_grid_tree = {'max_depth': [3, 5], 'min_samples_split': [2, 5]}
grid_search_tree = GridSearchCV(tree_clf, param_grid_tree, cv=kfold, scoring='accuracy')
grid_search_tree.fit(X, y)

best_tree_clf = grid_search_tree.best_estimator_
scores_tree = cross_val_score(best_tree_clf, X, y, cv=kfold, scoring='accuracy')

print(f"Árvore de Decisão - Melhor Parâmetro: {grid_search_tree.best_params_}")
print(f"Árvore de Decisão - Accuracy: Média = {scores_tree.mean():.4f}, Desvio Padrão = {scores_tree.std():.4f}")

# Visualização da Árvore de Decisão
plt.figure(figsize=(20, 10))
plot_tree(best_tree_clf, filled=True, feature_names=X.columns, class_names=[str(c) for c in best_tree_clf.classes_], rounded=True)
plt.title('Árvore de Decisão - Melhor Modelo')
plt.show()


# 1.2. SVM
svc_clf = SVC(random_state=42)
param_grid_svc = {'kernel': ['linear','poly', 'rbf', 'sigmoid'], 'C': [0.1, 1, 10, 100]}
grid_search_svc = GridSearchCV(svc_clf, param_grid_svc, cv=kfold, scoring='accuracy')
grid_search_svc.fit(X, y)

best_svc_clf = grid_search_svc.best_estimator_
scores_svc = cross_val_score(best_svc_clf, X, y, cv=kfold, scoring='accuracy')

print(f"SVM - Melhor Kernel: {grid_search_svc.best_params_['kernel']}")
print(f"SVM - Accuracy: Média = {scores_svc.mean():.4f}, Desvio Padrão = {scores_svc.std():.4f}")

# 1.3. Rede Neuronal
def create_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

nn_clf = KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=0, input_dim=X.shape[1])
scores_nn = cross_val_score(nn_clf, X, y, cv=kfold, scoring='accuracy')

print(f"Rede Neuronal - Accuracy: Média = {scores_nn.mean():.4f}, Desvio Padrão = {scores_nn.std():.4f}")

# Plot da curva de perda da melhor rede neural treinada
nn_clf.fit(X, y)
plt.plot(nn_clf.model_.history.history['loss'])
plt.title('Rede Neuronal - Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# 1.4. K-vizinhos-mais-próximos
knn_clf = KNeighborsClassifier()
param_grid_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
grid_search_knn = GridSearchCV(knn_clf, param_grid_knn, cv=kfold, scoring='accuracy')
grid_search_knn.fit(X, y)

best_knn_clf = grid_search_knn.best_estimator_
scores_knn = cross_val_score(best_knn_clf, X, y, cv=kfold, scoring='accuracy')

print(f"KNN - Melhor K: {grid_search_knn.best_params_['n_neighbors']}")
print(f"KNN - Accuracy: Média = {scores_knn.mean():.4f}, Desvio Padrão = {scores_knn.std():.4f}")

# Comparação dos Resultados
results = {
    'Modelo': ['Árvore de Decisão', 'SVM', 'Rede Neuronal', 'KNN'],
    'Accuracy Média': [scores_tree.mean(), scores_svc.mean(), scores_nn.mean(), scores_knn.mean()],
    'Accuracy Desvio Padrão': [scores_tree.std(), scores_svc.std(), scores_nn.std(), scores_knn.std()]
}

results_df = pd.DataFrame(results)
print(results_df)

# 1. a) Teste de significância estatística entre os dois melhores modelos
best_models_scores = {'Árvore de Decisão': scores_tree, 'SVM': scores_svc, 'Rede Neuronal': scores_nn, 'KNN': scores_knn}
best_models = sorted(best_models_scores, key=lambda k: best_models_scores[k].mean(), reverse=True)[:2]
scores_best_model_1 = best_models_scores[best_models[0]]
scores_best_model_2 = best_models_scores[best_models[1]]

t_stat, p_value = ttest_rel(scores_best_model_1, scores_best_model_2)
print(f"P-valor ({best_models[0]} vs {best_models[1]}): {p_value:.4f}")

if p_value < 0.05:
    print("A diferença entre os modelos é estatisticamente significativa.")
else:
    print("A diferença entre os modelos não é estatisticamente significativa.")
    
# 1. b) Comparação dos resultados dos modelos anteriores
def calculate_metrics(model, X, y):
    y_pred = cross_val_predict(model, X, y, cv=kfold)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y, y_pred, average='weighted', zero_division=1)  # Sensitivity
    cm = confusion_matrix(y, y_pred)
    specificity = np.mean(np.diag(cm) / (np.sum(cm, axis=1) + 1e-10))
    return accuracy, f1, recall, specificity, cm

# Calcular métricas para os modelos
accuracy_tree, f1_tree, recall_tree, specificity_tree, cm_tree = calculate_metrics(best_tree_clf, X, y)
accuracy_svc, f1_svc, recall_svc, specificity_svc, cm_svc = calculate_metrics(best_svc_clf, X, y)
accuracy_nn, f1_nn, recall_nn, specificity_nn, cm_nn = calculate_metrics(nn_clf, X, y)
accuracy_knn, f1_knn, recall_knn, specificity_knn, cm_knn = calculate_metrics(best_knn_clf, X, y)

# Exibir as métricas
print(f"Árvore de Decisão - Accuracy: {accuracy_tree:.4f}, F1: {f1_tree:.4f}, Sensitivity: {recall_tree:.4f}, Specificity: {specificity_tree:.4f}")
print(f"SVM - Accuracy: {accuracy_svc:.4f}, F1: {f1_svc:.4f}, Sensitivity: {recall_svc:.4f}, Specificity: {specificity_svc:.4f}")
print(f"Rede Neuronal - Accuracy: {accuracy_nn:.4f}, F1: {f1_nn:.4f}, Sensitivity: {recall_nn:.4f}, Specificity: {specificity_nn:.4f}")
print(f"KNN - Accuracy: {accuracy_knn:.4f}, F1: {f1_knn:.4f}, Sensitivity: {recall_knn:.4f}, Specificity: {specificity_knn:.4f}")

# 1. c) Seleção de atributos

# Seleção de atributos com SelectKBest
k = 10  # Número de atributos a selecionar
selector = SelectKBest(f_classif, k=k)
X_selected = selector.fit_transform(X, y)

# Atributos selecionados
selected_features = X.columns[selector.get_support()]
print(f"Atributos selecionados: {selected_features}")

# Árvores de Decisão
tree_clf_selected = DecisionTreeClassifier(random_state=42)
param_grid_tree_selected = {'max_depth': [3, 5], 'min_samples_split': [2, 5]}
grid_search_tree_selected = GridSearchCV(tree_clf_selected, param_grid_tree_selected, cv=kfold, scoring='accuracy')
grid_search_tree_selected.fit(X_selected, y)

best_tree_clf_selected = grid_search_tree_selected.best_estimator_
scores_tree_selected = cross_val_score(best_tree_clf_selected, X_selected, y, cv=kfold, scoring='accuracy')

# SVM
svc_clf_selected = SVC(random_state=42)
param_grid_svc_selected = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [0.1, 1, 10, 100]}
grid_search_svc_selected = GridSearchCV(svc_clf_selected, param_grid_svc_selected, cv=kfold, scoring='accuracy')
grid_search_svc_selected.fit(X_selected, y)

best_svc_clf_selected = grid_search_svc_selected.best_estimator_
scores_svc_selected = cross_val_score(best_svc_clf_selected, X_selected, y, cv=kfold, scoring='accuracy')

# Rede Neuronal
def create_model_selected(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

nn_clf_selected = KerasClassifier(model=create_model_selected, epochs=100, batch_size=10, verbose=0, input_dim=X_selected.shape[1])
scores_nn_selected = cross_val_score(nn_clf_selected, X_selected, y, cv=kfold, scoring='accuracy')

# K-vizinhos-mais-próximos
knn_clf_selected = KNeighborsClassifier()
param_grid_knn_selected = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
grid_search_knn_selected = GridSearchCV(knn_clf_selected, param_grid_knn_selected, cv=kfold, scoring='accuracy')
grid_search_knn_selected.fit(X_selected, y)

best_knn_clf_selected = grid_search_knn_selected.best_estimator_
scores_knn_selected = cross_val_score(best_knn_clf_selected, X_selected, y, cv=kfold, scoring='accuracy')


# Comparação dos Resultados
results_selected = {
    'Modelo': ['Árvore de Decisão', 'SVM', 'Rede Neuronal', 'KNN'],
    'Accuracy Média': [scores_tree_selected.mean(), scores_svc_selected.mean(), scores_nn_selected.mean(), scores_knn_selected.mean()],
    'Accuracy Desvio Padrão': [scores_tree_selected.std(), scores_svc_selected.std(), scores_nn_selected.std(), scores_knn_selected.std()]
}

results_selected_df = pd.DataFrame(results_selected)
print(results_selected_df)

# Comparação das métricas dos modelos utilizando todos os atributos vs atributos selecionados
accuracy_tree_selected, f1_tree_selected, recall_tree_selected, specificity_tree_selected, cm_tree_selected = calculate_metrics(best_tree_clf_selected, X_selected, y)
accuracy_svc_selected, f1_svc_selected, recall_svc_selected, specificity_svc_selected, cm_svc_selected = calculate_metrics(best_svc_clf_selected, X_selected, y)
accuracy_nn_selected, f1_nn_selected, recall_nn_selected, specificity_nn_selected, cm_nn_selected = calculate_metrics(nn_clf_selected, X_selected, y)
accuracy_knn_selected, f1_knn_selected, recall_knn_selected, specificity_knn_selected, cm_knn_selected = calculate_metrics(best_knn_clf_selected, X_selected, y)

print(f"Árvore de Decisão (Atributos selecionados) - Accuracy: {accuracy_tree_selected:.4f}, F1: {f1_tree_selected:.4f}, Sensitivity: {recall_tree_selected:.4f}, Specificity: {specificity_tree_selected:.4f}")
print(f"SVM (Atributos selecionados) - Accuracy: {accuracy_svc_selected:.4f}, F1: {f1_svc_selected:.4f}, Sensitivity: {recall_svc_selected:.4f}, Specificity: {specificity_svc_selected:.4f}")
print(f"Rede Neuronal (Atributos selecionados) - Accuracy: {accuracy_nn_selected:.4f}, F1: {f1_nn_selected:.4f}, Sensitivity: {recall_nn_selected:.4f}, Specificity: {specificity_nn_selected:.4f}")
print(f"KNN (Atributos selecionados) - Accuracy: {accuracy_knn_selected:.4f}, F1: {f1_knn_selected:.4f}, Sensitivity: {recall_knn_selected:.4f}, Specificity: {specificity_knn_selected:.4f}")

# 2. Derivar Novos Preditores
data['Altura_Peso'] = data['Altura'] * data['Peso']
data['IMC_Hist_Obes'] = data['IMC'] * data['Historico_obesidade_familiar']

X_new_pred = data.drop(columns=['Label'])
y_new_pred = data['Label']

# Reavaliar os dois melhores modelos com os novos preditores
scores_best_model_1_new_pred = cross_val_score(best_tree_clf, X_new_pred, y_new_pred, cv=kfold, scoring='accuracy')
scores_best_model_2_new_pred = cross_val_score(best_svc_clf, X_new_pred, y_new_pred, cv=kfold, scoring='accuracy')

print(f"{best_models[0]} com os novos preditores - Accuracy: Média = {scores_best_model_1_new_pred.mean():.4f}, Desvio Padrão = {scores_best_model_1_new_pred.std():.4f}")
print(f"{best_models[1]} com os novos preditores - Accuracy: Média = {scores_best_model_2_new_pred.mean():.4f}, Desvio Padrão = {scores_best_model_2_new_pred.std():.4f}")

# Estatísticas de significância
t_stat_new_pred_1, p_value_new_pred_1 = ttest_rel(scores_best_model_1, scores_best_model_1_new_pred)
t_stat_new_pred_2, p_value_new_pred_2 = ttest_rel(scores_best_model_2, scores_best_model_2_new_pred)

# Resultados
print(f"P-valor (Modelo 1 com os novos preditores): {p_value_new_pred_1:.4f}")
print(f"P-valor (Modelo 2 com os novos preditores): {p_value_new_pred_2:.4f}")

if p_value_new_pred_1 < 0.05:
    print(f"A diferença para o Modelo 1 ({best_models[0]}) com os novos preditores é estatisticamente significativa.")
else:
    print(f"A diferença para o Modelo 1 ({best_models[0]}) com os novos preditores não é estatisticamente significativa.")

if p_value_new_pred_2 < 0.05:
    print(f"A diferença para o Modelo 2 ({best_models[1]}) com os novos preditores é estatisticamente significativa.")
else:
    print(f"A diferença para o Modelo 2 ({best_models[1]}) com os novos preditores não é estatisticamente significativa.")

# 3. Capacidade Preditiva do Atributo "Genero"
X_gender = data.drop(columns=['Genero'])
y_gender = data['Genero']

# Rede Neuronal para prever "Genero"
nn_gender_clf = KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=0, input_dim=X_gender.shape[1])
scores_nn_gender = cross_val_score(nn_gender_clf, X_gender, y_gender, cv=kfold, scoring='accuracy')

print(f"Rede Neuronal - Accuracy para prever 'Genero': Média = {scores_nn_gender.mean():.4f}, Desvio Padrão = {scores_nn_gender.std():.4f}")

# SVM para prever "Genero"
best_svc_gender_clf = grid_search_svc.best_estimator_
scores_svc_gender = cross_val_score(best_svc_gender_clf, X_gender, y_gender, cv=kfold, scoring='accuracy')

print(f"SVM - Accuracy para prever 'Genero': Média = {scores_svc_gender.mean():.4f}, Desvio Padrão = {scores_svc_gender.std():.4f}")

# 3. b) Teste de significância estatística entre os dois melhores modelos
best_models_scores_gender = {'Rede Neuronal': scores_nn_gender, 'SVM': scores_svc_gender}
best_models_gender = sorted(best_models_scores_gender, key=lambda k: best_models_scores_gender[k].mean(), reverse=True)[:2]
scores_best_model_1_gender = best_models_scores_gender[best_models_gender[0]]
scores_best_model_2_gender = best_models_scores_gender[best_models_gender[1]]

t_stat_gender, p_value_gender = ttest_rel(scores_best_model_1_gender, scores_best_model_2_gender)
print(f"P-valor ({best_models_gender[0]} vs {best_models_gender[1]}): {p_value_gender:.4f}")

if p_value_gender < 0.05:
    print("A diferença entre os modelos é estatisticamente significativa.")
else:
    print("A diferença entre os modelos não é estatisticamente significativa.")
    
# 3 c) Comparação dos resultados dos modelos nos critérios: Accuracy, Sensitivity, Specificity e F1

def calculate_metrics(model, X_gender, y_gender):
    y_pred = cross_val_predict(model, X_gender, y_gender, cv=kfold)
    accuracy = accuracy_score(y_gender, y_pred)
    f1 = f1_score(y_gender, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_gender, y_pred, average='weighted', zero_division=1)  # Sensitivity
    cm = confusion_matrix(y_gender, y_pred)
    specificity = np.mean(np.diag(cm) / (np.sum(cm, axis=1) + 1e-10))
    return accuracy, f1, recall, specificity, cm

accuracy_svc, f1_svc, recall_svc, specificity_svc, cm_svc = calculate_metrics(best_svc_gender_clf, X_gender, y_gender)
accuracy_nn, f1_nn, recall_nn, specificity_nn, cm_nn = calculate_metrics(nn_gender_clf, X_gender, y_gender)

metrics = {
    'Modelo': ['SVM', 'Rede Neuronal'],
    'Accuracy': [accuracy_svc, accuracy_nn],
    'Sensitivity': [recall_svc, recall_nn],
    'Specificity': [specificity_svc, specificity_nn],
    'F1 Score': [f1_svc, f1_nn]
}

metrics_df = pd.DataFrame(metrics)
print(metrics_df)

best_model_idx = metrics_df[['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']].mean(axis=1).idxmax()
best_model = metrics_df.loc[best_model_idx, 'Modelo']
print(f"O modelo com melhor desempenho geral é: {best_model}")
