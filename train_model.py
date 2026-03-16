import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import joblib

# ── 1. Load & clean data 
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df = df.dropna()

# Fix duplicate column names
cols = df.columns.tolist()
cols[21] = 'RainToday'
cols[22] = 'RainTomorrow'
df.columns = cols

# Rename to match our feature engineering logic 
df = df.rename(columns={'RainToday': 'RainYesterday', 'RainTomorrow': 'RainToday'})

# Filter to Melbourne area only
df = df[df.Location.isin(['Melbourne', 'MelbourneAirport', 'Watsonia'])]

# ── 2. Feature engineering 
def date_to_season(date):
    month = date.month
    if month in [12, 1, 2]:   return 'Summer'
    elif month in [3, 4, 5]:  return 'Autumn'
    elif month in [6, 7, 8]:  return 'Winter'
    else:                      return 'Spring'

df['Date'] = pd.to_datetime(df['Date'])
df['Season'] = df['Date'].apply(date_to_season)
df = df.drop(columns='Date')

# ── 3. Split 
X = df.drop(columns='RainToday')
y = df['RainToday']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── 4. Feature lists 
numeric_features    = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# ── 5. Transformers 
numeric_transformer     = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer,     numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ── 6. Pipeline + GridSearch 
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',   RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators':    [50, 100],
    'classifier__max_depth':       [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

print(f"\nBest params : {grid_search.best_params_}")
print(f"CV score    : {grid_search.best_score_:.3f}")
print(f"Test score  : {grid_search.score(X_test, y_test):.3f}")

# ── 7. Save model + feature lists 
joblib.dump({
    'model':               grid_search,
    'numeric_features':    numeric_features,
    'categorical_features': categorical_features,
    'feature_order':       X_train.columns.tolist()
}, 'model.pkl')

print("\n✅  model.pkl saved successfully!")