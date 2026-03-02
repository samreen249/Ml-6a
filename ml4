import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score 
data = { 
    'Age': ['Young','Young','Middle','Old','Old','Middle','Young','Old'], 
    'Income': ['High','High','Medium','Low','Low','Medium','Medium','High'], 
    'Student': ['No','No','No','Yes','Yes','Yes','No','Yes'], 
    'Credit': ['Fair','Excellent','Fair','Fair','Excellent','Excellent','Fair','Excellent'], 
    'Buy': ['No','No','Yes','Yes','No','Yes','Yes','No'] 
} 
 
df = pd.DataFrame(data) 
print("Original Dataset:\n", df) 
le = LabelEncoder() 
for col in df.columns: 
    df[col] = le.fit_transform(df[col]) 
print("\nEncoded Dataset:\n", df) 
X = df.drop('Buy', axis=1) 
y = df['Buy'] 
 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.25, random_state=42 
    ) 
 
# ------------------------------------------------ 
# Step 5: Decision Tree WITHOUT parameter tuning 
# ------------------------------------------------ 
dt_before = DecisionTreeClassifier(random_state=42) 
dt_before.fit(X_train, y_train) 
 
y_pred_before = dt_before.predict(X_test) 
accuracy_before = accuracy_score(y_test, y_pred_before) 
 
print("\nAccuracy BEFORE parameter tuning:", accuracy_before) 
 
# ------------------------------------------------ 
# Step 6: Decision Tree WITH parameter tuning 
# ------------------------------------------------ 
dt_after = DecisionTreeClassifier( 
      criterion='entropy', 
    max_depth=3, 
    random_state=42 
) 
 
dt_after.fit(X_train, y_train) 
 
y_pred_after = dt_after.predict(X_test) 
accuracy_after = accuracy_score(y_test, y_pred_after) 
 
print("Accuracy AFTER parameter tuning:", accuracy_after) 
