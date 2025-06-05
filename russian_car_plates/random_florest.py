import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.linear_model import LinearRegression
import plotly.express as px
from pathlib import Path

base_dir = Path(__file__).parent

train_data = pd.read_csv(base_dir / 'train_enriched.csv')
test_data = pd.read_csv(base_dir / 'test_enriched.csv')

input_cols = train_data.columns.difference(['id', 'price', 'date', 'plate']).tolist()
target_cols = 'price'

train_inputs, train_target = train_data[input_cols].copy(), train_data[target_cols].copy()
test_inputs = test_data[input_cols].copy()

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist() 

#Scale numeric features 
scaler = MinMaxScaler().fit(train_data[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# One-hot encoding das colunas categóricas
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
train_encoded = encoder.fit_transform(train_inputs[categorical_cols])
test_encoded = encoder.transform(test_inputs[categorical_cols])

# Criar DataFrames com as colunas codificadas corretamente
encoded_cols = encoder.get_feature_names_out(categorical_cols)
train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=train_inputs.index)
test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=test_inputs.index)

# Remover colunas categóricas antigas e adicionar as novas codificadas
train_inputs = pd.concat([train_inputs.drop(columns=categorical_cols), train_encoded_df], axis=1)
test_inputs = pd.concat([test_inputs.drop(columns=categorical_cols), test_encoded_df], axis=1)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
#score 0.8362772989304912
#                    feature  importance
# 2            plate_numbers    0.303128
# 3             plate_region    0.091427
# 4       significance_level    0.042539
# 17    plate_first_letter_X    0.042164
# 97   plate_last_letters_MM    0.030501
# 43   plate_last_letters_CA    0.030096
# 8     plate_first_letter_B    0.029998
# 13    plate_first_letter_M    0.027877
# 111  plate_last_letters_OP    0.022177
# 19   plate_last_letters_AA    0.021253
# RMSE no conjunto de validação: 725604.70
# R² no conjunto de validação: 0.84

# Gerar o gráfico de dispersão
# fig = px.scatter(train_data, x='plate_numbers', y='price', title='Preço x Número da Placa')

# fig.show()


# model = LinearRegression()
#0.06449600007799172  - linear simples


model.fit(train_inputs,train_target)



train_pred = model.predict(train_inputs)
# print(train_pred)

print(model.score(train_inputs, train_target))
rmse = np.sqrt(mean_squared_error(train_target, train_pred))
r2 = r2_score(train_target, train_pred)

#Isso é para a Floresta
importance_df = pd.DataFrame({
    'feature': train_inputs.columns ,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# print(importance_df.head(20))

# print(f"RMSE no conjunto de validação: {rmse:.2f}")
# print(f"R² no conjunto de validação: {r2:.2f}") 


# model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
#Com as datas
# 0.8439365647180191
# RMSE no conjunto de validação: 708428.82
# R² no conjunto de validação: 0.84 


# 0.8439365647180191
#                               feature  importance
# 2                       plate_numbers    0.301689
# 3                        plate_region    0.090173
# 17               plate_first_letter_X    0.043720
# 4                  significance_level    0.042820
# 97              plate_last_letters_MM    0.031133
# 43              plate_last_letters_CA    0.030038
# 8                plate_first_letter_B    0.029281
# 13               plate_first_letter_M    0.025995
# 111             plate_last_letters_OP    0.022251
# 19              plate_last_letters_AA    0.020979
# 7                plate_first_letter_A    0.018959
# 18               plate_first_letter_Y    0.018749
# 198                region_name_Moscow    0.014457
# 10               plate_first_letter_E    0.014277
# 190        region_name_Krasnodar Krai    0.013675
# 216  region_name_Republic of Dagestan    0.013474
# 191      region_name_Krasnoyarsk Krai    0.012444
# 149             plate_last_letters_XX    0.012179
# 54              plate_last_letters_CY    0.011174
# 110             plate_last_letters_OO    0.010962
# RMSE no conjunto de validação: 708428.82
# R² no conjunto de validação: 0.84
# PS C:\Users\HP\Documents\AI-courses> 

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  # Para evitar divisão por zero
    return 100 * np.mean(diff)

# Calcular o SMAPE no seu modelo
smape_value = smape(train_target, train_pred)
print(f"SMAPE no conjunto de validação: {smape_value:.2f}")

def prepare_submission(model, X_test):
    # Fazer previsões no conjunto de teste
    test_pred = model.predict(X_test)
    
    # Criar arquivo de submissão
    submission = pd.DataFrame({
        'id': test_data['id'],
        'price': test_pred
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Arquivo de submissão criado: submission.csv")

prepare_submission(model, test_inputs)