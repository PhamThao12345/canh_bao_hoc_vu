import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from scipy.sparse import hstack, csr_matrix

# --- 1. NẠP MÔ HÌNH (LOAD ARTIFACTS) ---
@st.cache_resource # Cache để chỉ load 1 lần khi mở app
def load_artifacts():
    with open('student_model_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

artifacts = load_artifacts()
model = artifacts['model']
tfidf = artifacts['tfidf']
label_encoders = artifacts['label_encoders']
tabular_columns = artifacts['tabular_columns']

# --- 2. CÁC HÀM TIỀN XỬ LÝ ---
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

# --- 3. GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Hệ thống Cảnh báo Học vụ", layout="centered")
st.title("🎓 Ứng dụng Dự đoán Cảnh báo Học vụ")
st.markdown("Mô hình **LightGBM** đã được huấn luyện trước. Bạn chỉ cần tải file `test.csv` lên để xem dự đoán.")

uploaded_file = st.file_uploader("Tải lên tập kiểm tra (test.csv)", type=["csv"])

if uploaded_file is not None:
    st.success("Tải file thành công! Đang tiến hành dự đoán...")
    
    # Đọc data
    test_df = pd.read_csv(uploaded_file)
    test_ids = test_df['Student_ID'] if 'Student_ID' in test_df.columns else np.arange(len(test_df))
    
    with st.spinner('Đang xử lý dữ liệu...'):
        # Biến đổi dữ liệu mới tương tự như lúc train
        test_df['English_Level'] = test_df['English_Level'].apply(clean_english_level)
        test_df['Admission_Mode'] = test_df['Admission_Mode'].apply(clean_admission_mode)
        
        # Số
        num_cols = ['Age', 'Tuition_Debt', 'Count_F', 'Training_Score_Mixed'] + [col for col in test_df.columns if 'Att_Subject' in col]
        for col in num_cols:
            if col in test_df.columns:
                test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(-1)
            else:
                test_df[col] = -1 # Nếu thiếu cột số thì điền -1
                
        # Categorical
        for col, le in label_encoders.items():
            if col in test_df.columns:
                test_df[col] = test_df[col].astype(str).fillna('unknown')
                # Xử lý nhãn mới (chưa từng thấy lúc train)
                # Thay bằng 'unknown' nếu nhãn không có trong classes_
                known_classes = set(le.classes_)
                test_df[col] = test_df[col].apply(lambda x: x if x in known_classes else 'unknown')
                
                # Transform
                test_df[col] = le.transform(test_df[col])
            else:
                test_df[col] = -1

        # Văn bản
        test_df['Combined_Text'] = test_df['Advisor_Notes'].apply(clean_text) + " " + test_df['Personal_Essay'].apply(clean_text)
        text_features_test = tfidf.transform(test_df['Combined_Text'])
        
        # Sắp xếp đúng thứ tự cột Tabular
        X_test_tabular = test_df[tabular_columns]
        X_test_sparse = csr_matrix(X_test_tabular.values)
        
        # Nối feature Tabular và Text
        X_test_final = hstack([X_test_sparse, text_features_test]).tocsr()
        
        # DỰ ĐOÁN
        preds = model.predict(X_test_final)
        
        # Format kết quả
        submission = pd.DataFrame({
            'Student_ID': test_ids,
            'Academic_Status': preds
        })
        
        # Mapping label sang Text để hiển thị đẹp hơn trên Web
        status_map = {0: '0 - Normal', 1: '1 - Academic Warning', 2: '2 - Dropout'}
        display_df = submission.copy()
        display_df['Academic_Status_Label'] = display_df['Academic_Status'].map(status_map)
        
    st.write("### 📊 Trích xuất kết quả dự đoán (5 dòng đầu):")
    st.dataframe(display_df.head(), use_container_width=True)
    
    # Download
    csv = submission.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải file submission.csv (Để nộp Kaggle)",
        data=csv,
        file_name='submission.csv',
        mime='text/csv',
    )
