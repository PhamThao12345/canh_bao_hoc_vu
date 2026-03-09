# Đọc dữ liệu
train_df = pd.read_csv('train.csv')

def feature_engineering(df):
    df = df.copy()
    # Chuyển văn bản thành độ dài chuỗi (Baseline đơn giản)
    df['Advisor_Notes_len'] = df['Advisor_Notes'].astype(str).apply(len)
    df['Personal_Essay_len'] = df['Personal_Essay'].astype(str).apply(len)
    
    # Bỏ các cột văn bản thô vì model dạng cây (Random Forest) không đọc được chữ trực tiếp
    cols_to_drop = ['Advisor_Notes', 'Personal_Essay', 'Current_Address', 'Hometown']
    if 'Academic_Status' in df.columns: cols_to_drop.append('Academic_Status')
    if 'Student_ID' in df.columns: cols_to_drop.append('Student_ID')
        
    return df.drop(columns=cols_to_drop, errors='ignore')

X = feature_engineering(train_df)
y = train_df['Academic_Status']

# Lọc ra danh sách các cột là Số và các cột là Chữ (Categorical)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Xử lý cột Số: Điền giá trị thiếu (Median) và Chuẩn hóa dữ liệu (Scaler)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Xử lý cột Chữ: Điền giá trị xuất hiện nhiều nhất và Biến đổi thành vector 0-1 (OneHot)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Gộp 2 bộ xử lý lại
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Nối phần tiền xử lý và thuật toán Random Forest vào một Pipeline duy nhất
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

model.fit(X, y) # Bắt đầu cho máy học từ dữ liệu

# Lưu mô hình ra file dạng byte (pickle)
with open('student_dropout_model.pkl', 'wb') as f:
    pickle.dump(model, f)
