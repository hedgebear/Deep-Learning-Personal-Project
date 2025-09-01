import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- INICIALIZAÇÃO ---

# Passo 1: Recebe as entradas em formato de matriz
print("Passo 1: Carregando e preparando os dados...")
# Carrega o dataset a partir do arquivo CSV
# Certifique-se que o arquivo 'diabetes.csv' está na mesma pasta
data = pd.read_csv('C:\\GitHub\\Deep-Learning-Personal-Project\\prever_diabetes\\diabetes.csv')

# Separa as features (variáveis de entrada, X) do alvo (variável de saída, y)
X = data.drop('Outcome', axis=1) # Todas as colunas, exceto 'Outcome'
y = data['Outcome']              # Apenas a coluna 'Outcome'

# Divide os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza os dados. Redes neurais funcionam melhor com dados em escalas semelhantes.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Formato dos dados de treino (matriz de entrada): {X_train.shape}")
print("-" * 30)


# Passo 2: Inicializa os pesos (incluindo bias) com valores aleatórios
print("Passo 2: Construindo o modelo e inicializando os pesos...")
# Keras faz a inicialização dos pesos automaticamente ao criar as camadas.
model = keras.Sequential([
    # Camada de entrada (implícita) e primeira camada escondida com 12 neurônios
    # input_shape deve corresponder ao número de features (8 colunas em X)
    keras.layers.Dense(12, input_shape=(8,), activation='relu'),
    
    # Segunda camada escondida com 8 neurônios
    keras.layers.Dense(8, activation='relu'),
    
    # Camada de saída com 1 neurônio. 'sigmoid' para classificação binária (saída entre 0 e 1)
    keras.layers.Dense(1, activation='sigmoid')
])

# Exibe um resumo da arquitetura do modelo
model.summary()
print("-" * 30)

# --- PROPAGAÇÃO E RETROPROPAGAÇÃO (O LOOP DE TREINO) ---

# Aqui, configuramos as ferramentas para os passos 6 a 9.
# O 'optimizer' cuida dos passos 7, 8 e 9 (cálculo e aplicação dos pesos ajustados).
# A 'loss' é a função para o passo 6 (cálculo do erro).
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Passo 3: Repete os passos 4 a 9 até que o erro seja minimizado
print("Passo 3 a 9: Iniciando o processo de treinamento...")
# A função 'fit' executa o loop de treinamento (épocas).
# Em cada época, ela realiza a propagação à frente e a retropropagação.
history = model.fit(
    X_train,
    y_train,
    epochs=150,           # Número de vezes que o modelo verá todo o dataset
    batch_size=10,        # Número de amostras por atualização de gradiente
    validation_split=0.2, # Usa 20% dos dados de treino para validação
    verbose=0             # Define verbose=0 para não poluir a saída. Mude para 1 para ver o progresso.
)
print("Treinamento concluído!")
print("-" * 30)


# --- AVALIAÇÃO E USO ---

print("Avaliando o modelo treinado com os dados de teste...")
# Avalia o desempenho do modelo nos dados que ele nunca viu antes
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")

# Exemplo de como usar o modelo para fazer uma nova previsão
# Pegamos a primeira amostra do conjunto de teste para demonstrar
print("\nFazendo uma previsão em uma nova amostra...")
primeira_amostra = X_test[0]
previsao_bruta = model.predict(primeira_amostra.reshape(1, -1))
previsao_final = (previsao_bruta > 0.5).astype("int32")

print(f"Entrada da amostra (normalizada): {primeira_amostra}")
print(f"Saída da rede (probabilidade): {previsao_bruta[0][0]:.4f}")
print(f"Previsão final (0 = Não diabético, 1 = Diabético): {previsao_final[0][0]}")
print(f"Valor real da amostra: {y_test.iloc[0]}")