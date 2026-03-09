import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pickle

# 1. Đọc dữ liệu
print("Đang đọc dữ liệu...")
train_df = pd.read_csv('train.csv')

# 2. Hàm trích xuất đặc trưng (Feature Engineering)
def feature_engineering(df):
    df_transformed = df.copy()
    
    # Biến văn bản thành độ dài chuỗi để mô hình dễ đọc
    df_transformed['Advisor_Notes_len'] = df_transformed['Advisor_Notes'].astype(str).apply(len)
    df_transformed['Personal_Essay_len'] = df_transformed['Personal_Essay'].astype(str).apply(len)
    
    # Xóa các cột văn bản thô và các cột không cần thiết
    cols_to_drop = ['Advisor_Notes', 'Personal_Essay', 'Current_Address', 'Hometown']
    if 'Academic_Status' in df_transformed.columns:
        cols_to_drop.append('Academic_Status')
    if 'Student_ID' in df_transformed.columns:
        cols_to_drop.append('Student_ID')
        
    return df_transformed.drop(columns=cols_to_drop, errors='ignore')

# Tách features (X) và target (y)
X = feature_engineering(train_df)
y = train_df['Academic_Status']

# 3. Định nghĩa các cột Số và Chữ
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# 4. Pipeline Tiền xử lý
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. Khởi tạo Mô hình
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# 6. Huấn luyện mô hình
print("Đang huấn luyện mô hình (Quá trình này có thể mất vài chục giây)...")
model.fit(X, y)

# Đánh giá nhanh trên tập train
y_pred = model.predict(X)
print(f"Huấn luyện xong! Train Macro F1-Score: {f1_score(y, y_pred, average='macro'):.4f}")

# 7. Lưu mô hình (Serialize)
with open('student_dropout_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Đã lưu mô hình thành công vào file: student_dropout_model.pkl")
