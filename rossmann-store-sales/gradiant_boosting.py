import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

import os 

# Obtém o diretório do arquivo atual
path = os.path.dirname(os.path.abspath(__file__))

# Junta o caminho do diretório com os nomes dos arquivos
ross_df = pd.read_csv(os.path.join(path, 'train.csv'))
test_df = pd.read_csv(os.path.join(path, 'test.csv'))
submission_df = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
store_df = pd.read_csv(os.path.join(path, 'store.csv'))

#Unir os dataframes de treino e teste com o dataframe de lojas
merged_df = ross_df.merge(store_df, how='left', on='Store') 
merged_test_df = test_df.merge(store_df, how='left', on='Store')

#Feature Engineering - datas
def splite_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Week'] = df['Date'].dt.isocalendar().week
    return df

merged_df = splite_date(merged_df)
merged_test_df = splite_date(merged_test_df)

#Feature Engineering - excluindo lojas fechadas 
merged_df = merged_df[merged_df['Open'] == 1].copy() 

#Feature Engineering - meses de competição
def comp_months(df):
    df['Competition_months'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['Competition_months'] = df['Competition_months'].map(lambda x: x if x > 0 else 0).fillna(0)
    return df

merged_df = comp_months(merged_df)
merged_test_df = comp_months(merged_test_df)    

#Feature Engineering - verificar promoção ativa em rows
def check_promo_month(row):
    month2str = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                  7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    try: 
        months = str(row['PromoInterval']).split(',')
        if pd.notnull(row['Promo2Open']) and month2str.get(row['Month']) in months:
            return 1
        else:
            return 0
    except:
        return 0 
#Feature Engineering - início promoção2 e promoção2 ativa
def promo_cols(df):
    # Calcula semanas desde início da promoção
    df['Promo2Open'] = 52 * (df['Year'] - df['Promo2SinceYear']) + (df['Week'] - df['Promo2SinceWeek'])
    df['Promo2Open'] = df['Promo2Open'].apply(lambda x: x if x > 0 else 0).fillna(0) * df['Promo2']

    # Determina se é um mês em que a Promo2 está ativa (usando nomes dos meses em PromoInterval)
    df['IsPromo2Month'] = df.apply(check_promo_month, axis=1) * df['Promo2']
    
    return df


merged_df = promo_cols(merged_df)
merged_test_df = promo_cols(merged_test_df)


#Selecionando as colunas de entrada e o alvo
input_cols = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'Week', 'Promo',
              'StoreType', 'Assortment', 'CompetitionDistance', 'Competition_months',  
              'Promo2Open', 'IsPromo2Month', 'Promo2', 'SchoolHoliday', 'StateHoliday']
target_col = 'Sales'

inputs = merged_df[input_cols].copy()
target = merged_df[target_col].copy()

test_inputs = merged_test_df[input_cols].copy() 

#Separando as colunas numéricas e categóricas
numerical_cols = ['Store', 'Promo', 'SchoolHoliday', 'CompetitionDistance', 
                  'Competition_months', 'Promo2Open', 'Promo2', 'IsPromo2Month',
                  'Year', 'Month', 'Day', 'Week',  ]
categorical_cols = ['StoreType', 'DayOfWeek', 'StateHoliday', 'Assortment']

# print(inputs[numerical_cols].isna().sum())
# print(test_inputs[numerical_cols].isna().sum())
max_distance = inputs['CompetitionDistance'].max()
inputs['CompetitionDistance'] = inputs['CompetitionDistance'].fillna(max_distance*2)
test_inputs['CompetitionDistance'] = test_inputs['CompetitionDistance'].fillna(max_distance*2)

# Normalizando as colunas numéricas
scaler = MinMaxScaler().fit(inputs[numerical_cols])
inputs[numerical_cols] = scaler.transform(inputs[numerical_cols])
test_inputs[numerical_cols] = scaler.transform(test_inputs[numerical_cols])

# One-hot encoding das colunas categóricas
# Convertendo 'StateHoliday' para string para evitar problemas com OneHotEncoder
inputs['StateHoliday'] = inputs['StateHoliday'].astype(str)
test_inputs['StateHoliday'] = test_inputs['StateHoliday'].astype(str)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
train_encoded = encoder.fit_transform(inputs[categorical_cols])
test_encoded = encoder.transform(test_inputs[categorical_cols])

# Criar DataFrames com as colunas codificadas corretamente
encoded_cols = encoder.get_feature_names_out(categorical_cols)
train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=inputs.index)
test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=test_inputs.index)

train_inputs = pd.concat([inputs.drop(columns=categorical_cols), train_encoded_df], axis=1)
test_inputs = pd.concat([test_inputs.drop(columns=categorical_cols), test_encoded_df], axis=1)


def rmse(a, b):
    return root_mean_squared_error(a, b)

def train_and_evaluate(train_inputs, train_target, validation_inputs, validation_target, **params):
    model = XGBRegressor(random_state=40, **params)
    model.fit(train_inputs, train_target) 
    preds = model.predict(validation_inputs)
    rmsevalidation = rmse(preds, validation_target)
    rmse_train = rmse(model.predict(train_inputs), train_target)
    return model, rmsevalidation, rmse_train 

models = []


def predict_average(models, inputs):
    return np.mean([model.predict(inputs) for model in models], axis=0) 

def test_params_kfold(n_splits, **params):
    train_rmse, val_rmse, models = [], [], []
    kfold = KFold(n_splits)
    for train_index, val_index in kfold.split(train_inputs):
        X_train, y_train = train_inputs.iloc[train_index], target.iloc[train_index]
        X_val, y_val = train_inputs.iloc[val_index], target.iloc[val_index]
        
        model, val_rmse_, train_rmse_ = train_and_evaluate(
            X_train, y_train, X_val, y_val, **params)
        models.append(model)
        train_rmse.append(train_rmse_)
        val_rmse.append(val_rmse_)
    print(f"Train RMSE: {np.mean(train_rmse)}, Validation RMSE: {np.mean(val_rmse)}")
    return models

def test_params(**params):
    X_train, X_val, y_train, y_val = train_test_split(train_inputs, target, test_size=0.1)
    model = XGBRegressor(random_state=40, **params)
    model.fit(X_train, y_train)
    train_rmse = rmse(model.predict(X_train), y_train)
    val_rmse = rmse(model.predict(X_val), y_val)
    print(f"Train RMSE: {train_rmse}, Validation RMSE: {val_rmse}")
    

final_model = XGBRegressor(
    n_estimators=200, 
    max_depth=10, 
    learning_rate=0.2,
    
)
final_model.fit(train_inputs, target) 
final_preds = final_model.predict(train_inputs)
print(rmse(final_preds, target))

test_preds = final_model.predict(test_inputs)

submission_df['Sales'] = test_preds
submission_df['Sales'] = submission_df['Sales'] * test_df.Open.fillna(1.)

submission_df.to_csv(os.path.join(path, 'submission.csv'), index=False)
medians = inputs[numerical_cols].median().to_dict()

def single_input(input_data):
    input_df = pd.DataFrame([input_data])

    # Verificação: loja fechada
    if 'Open' in input_df.columns and input_df['Open'].values[0] == 0:
        return 0

    # Merge com store_df para pegar informações da loja
    merged_input_df = input_df.merge(store_df, how='left', on='Store')

    # Feature engineering: data e promoções
    merged_input_df = splite_date(merged_input_df)
    merged_input_df = comp_months(merged_input_df)
    merged_input_df = promo_cols(merged_input_df)

    # Preencher CompetitionDistance com a mesma lógica do treino
    merged_input_df['CompetitionDistance'] = merged_input_df['CompetitionDistance'].fillna(max_distance * 2)

    # Preencher colunas numéricas ausentes com medianas
    for col in numerical_cols:
        if col not in merged_input_df.columns or pd.isna(merged_input_df[col].values[0]):
            merged_input_df[col] = medians.get(col, 0)

    # Garantir que colunas categóricas estejam presentes
    for col in categorical_cols:
        if col not in merged_input_df.columns:
            merged_input_df[col] = '0'  # valor neutro
    merged_input_df['StateHoliday'] = merged_input_df['StateHoliday'].astype(str)

    # Normalizar dados numéricos
    merged_input_df[numerical_cols] = scaler.transform(merged_input_df[numerical_cols])

    # One-hot encoding dos dados categóricos
    input_encoded = encoder.transform(merged_input_df[categorical_cols])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_cols, index=merged_input_df.index)

    # Juntar final
    final_input_df = pd.concat([merged_input_df.drop(columns=categorical_cols), input_encoded_df], axis=1)

    # Garantir que a ordem das colunas está correta (mesmo que no treino)
    final_input_df = final_input_df[train_inputs.columns]

    return final_model.predict(final_input_df)[0]


print('teste')
print(single_input({
    'Store': 2,
    'DayOfWeek': 4,
    'Date': '2015-09-30',
    'Open': 1,
    'StateHoliday': 'a',
    'SchoolHoliday': 0,
    'Promo': 1,
}))
# "Id","Store","DayOfWeek","Date","Open","Promo","StateHoliday","SchoolHoliday"
