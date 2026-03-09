import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Dự đoán Cảnh báo Học vụ", layout="wide")

# 1. Hàm tải mô hình (Giải nén ra Model và Danh sách cột)
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'student_dropout_model.pkl')
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Trả về thuật toán và danh sách đặc trưng (features)
    return model_data['model'], model_data['features']

model, saved_columns = load_model()

# 2. Hàm Tiền xử lý (Giống hệt trên Kaggle)
def preprocess_and_encode(df, train_cols):
    df_transformed = df.copy()
    
    df_transformed['Advisor_Notes_len'] = df_transformed['Advisor_Notes'].astype(str).apply(len)
    df_transformed['Personal_Essay_len'] = df_transformed['Personal_Essay'].astype(str).apply(len)
    
    cols_to_drop = ['Advisor_Notes', 'Personal_Essay', 'Current_Address', 'Hometown', 'Student_ID', 'Academic_Status']
    df_transformed = df_transformed.drop(columns=[c for c in cols_to_drop if c in df_transformed.columns], errors='ignore')
    
    num_cols = df_transformed.select_dtypes(include=['number']).columns
    cat_cols = df_transformed.select_dtypes(include=['object']).columns
    df_transformed[num_cols] = df_transformed[num_cols].fillna(-1)
    df_transformed[cat_cols] = df_transformed[cat_cols].fillna('Unknown')
    
    df_encoded = pd.get_dummies(df_transformed).astype(int)
    
    # Ép danh sách cột của file test tải lên phải khớp 100% với lúc train
    for col in train_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    return df_encoded[train_cols]

# 3. Giao diện Web
st.title("🎓 Hệ thống Dự đoán Cảnh báo Học vụ")
st.markdown("Dự báo trạng thái: **0 - Normal** | **1 - Warning** | **2 - Dropout**")

uploaded_file = st.file_uploader("Tải file dữ liệu sinh viên (CSV)", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.dataframe(df_input.head())

    if st.button("Chạy Dự Đoán", type="primary"):
        with st.spinner('AI đang phân tích dữ liệu...'):
            try:
                # 1. Tiền xử lý và căn chỉnh cột
                X_test = preprocess_and_encode(df_input, train_cols=saved_columns)
                
                # 2. Dự đoán
                preds = model.predict(X_test)
                
                # 3. Xuất kết quả
                status_mapping = {0: '0 - Normal', 1: '1 - Warning', 2: '2 - Dropout'}
                result_df = pd.DataFrame({
                    'Student_ID': df_input['Student_ID'] if 'Student_ID' in df_input.columns else [f"SV_{i}" for i in range(len(preds))],
                    'Academic_Status': preds,
                    'Label': [status_mapping[p] for p in preds]
                })
                
                st.success("Dự đoán thành công!")
                st.dataframe(result_df)
                
                # Nút tải file nộp Kaggle
                csv_data = result_df[['Student_ID', 'Academic_Status']].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Tải file nộp bài (submission.csv)",
                    data=csv_data,
                    file_name='submission.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Lỗi: {e}")
