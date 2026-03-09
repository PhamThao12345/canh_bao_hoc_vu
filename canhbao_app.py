import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

print("1. Đang đọc dữ liệu train.csv...")
train_df = pd.read_csv('train.csv')

# --- CÁC HÀM TIỀN XỬ LÝ ---
def clean_english_level(x):
    if pd.isna(x): return 'unknown'
    x = str(x).lower().strip()
    if 'ielts' in x: return 'ielts'
    if 'b1' in x: return 'b1'
    if 'b2' in x: return 'b2'
    if 'a2' in x: return 'a2'
    if 'toeic' in x: return 'toeic'
    return x

def clean_admission_mode(x):
    return 'unknown' if pd.isna(x) else str(x).lower().strip()

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

print("2. Đang làm sạch và tiền xử lý dữ liệu...")
train_df['English_Level'] = train_df['English_Level'].apply(clean_english_level)
train_df['Admission_Mode'] = train_df['Admission_Mode'].apply(clean_admission_mode)

# Xử lý missing values cho số
num_cols = ['Age', 'Tuition_Debt', 'Count_F', 'Training_Score_Mixed'] + [col for col in train_df.columns if 'Att_Subject' in col]
for col in num_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(-1)

# Xử lý Categorical (Lưu lại LabelEncoder để dùng cho App)
cat_cols = ['Gender', 'Hometown', 'Current_Address', 'Admission_Mode', 'English_Level', 'Club_Member']
label_encoders = {}
for col in cat_cols:
    train_df[col] = train_df[col].astype(str).fillna('unknown')
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le # Lưu lại dictionary

# Xử lý Text (Lưu lại TF-IDF)
train_df['Combined_Text'] = train_df['Advisor_Notes'].apply(clean_text) + " " + train_df['Personal_Essay'].apply(clean_text)
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
text_features = tfidf.fit_transform(train_df['Combined_Text'])

# Chuẩn bị X, y
target_col = 'Academic_Status'
y_train = train_df[target_col].values
X_tabular = train_df.drop(columns=['Student_ID', 'Advisor_Notes', 'Personal_Essay', 'Combined_Text', target_col])
X_train_sparse = csr_matrix(X_tabular.values)
X_train_final = hstack([X_train_sparse, text_features]).tocsr()

print("3. Đang huấn luyện mô hình LightGBM...")
model = LGBMClassifier(
    objective='multiclass',
    num_class=3,
    learning_rate=0.05,
    n_estimators=300,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_final, y_train)

print("4. Đang lưu mô hình bằng Pickle...")
# Đóng gói tất cả vào 1 dictionary để dễ quản lý
artifacts = {
    'model': model,
    'tfidf': tfidf,
    'label_encoders': label_encoders,
    'tabular_columns': X_tabular.columns.tolist() # Lưu lại tên cột để map đúng thứ tự trên app
}

with open('student_model_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("✅ Đã huấn luyện và lưu file 'student_model_artifacts.pkl' thành công!")
