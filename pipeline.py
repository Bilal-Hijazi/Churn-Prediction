import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score,precision_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib



data = pd.read_csv("hf://datasets/scikit-learn/churn-prediction/dataset.csv")
data= data.drop_duplicates()
data = data.drop('customerID',axis=1)
data = data.rename(columns= {'tenure': 'Tenure (Months)'})

cat_cols = ['PaymentMethod','PaperlessBilling','Contract','StreamingMovies','StreamingTV'
                                     ,'TechSupport','DeviceProtection','OnlineBackup','OnlineSecurity','InternetService','MultipleLines','PhoneService','Dependents','Partner','gender']


data['TotalCharges'] = pd.to_numeric(data['TotalCharges'].str.strip(), errors='coerce')
num_cols = ['Tenure (Months)', 'MonthlyCharges', 'TotalCharges']
data['Churn'] = data['Churn'].map({"Yes":1,"No":0})
data=data.dropna()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy= "median"))])    

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers= [
        ("num",numeric_transformer,num_cols),
        ("cat", categorical_transformer,cat_cols)
    ]
)

#full pipeline 

fp = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", RandomForestClassifier(n_estimators=400, max_depth=8 ,min_samples_split=20,
    min_samples_leaf=10,max_features='sqrt',random_state=42,class_weight='balanced')

)
])
X = data.drop('Churn',axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2 ,stratify=y)

fp.fit(X_train,y_train)

y_pred = fp.predict(X_test)



print(classification_report(y_test,y_pred))



joblib.dump(fp,"churn_model.pkl")