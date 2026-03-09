import streamlit as st
import pandas as pd
import pickle
import os

# Cấu hình trang web
st.set_page_config(page_title="Dự đoán Cảnh báo Học vụ", layout="wide")

# 1. Hàm tải mô hình (Sử dụng đường dẫn tuyệt đối để tránh lỗi FileNotFoundError)
@st.cache_resource
def load_model():
    # Lấy thư mục hiện tại đang chứa file canhbao_app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Nối với tên file model
    model_path = os.path.join(current_dir, 'student_dropout_model.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Khởi chạy tải mô hình
model = load_model()

# 2. Hàm Feature Engineering (Phải giống hệt với lúc Train)
def feature_engineering(df):
    df_transformed = df.copy()
    df_transformed['Advisor_Notes_len'] = df_transformed['Advisor_Notes'].astype(str).apply(len)
    df_transformed['Personal_Essay_len'] = df_transformed['Personal_Essay'].astype(str).apply(len)
    
    cols_to_drop = ['Student_ID', 'Advisor_Notes', 'Personal_Essay', 'Current_Address', 'Hometown', 'Academic_Status']
    return df_transformed.drop(columns=[c for c in cols_to_drop if c in df_transformed.columns], errors='ignore')

# 3. Giao diện Web
st.title("🎓 Hệ thống Dự đoán Cảnh báo Học vụ")
st.markdown("""
Ứng dụng sử dụng Học máy (Machine Learning) để dự báo tình trạng học tập của sinh viên:
* **0 - Normal:** Học tập bình thường.
* **1 - Academic Warning:** Thuộc diện cảnh báo học vụ.
* **2 - Dropout:** Nguy cơ thôi học cao.
""")

st.subheader("Tải lên dữ liệu sinh viên (File test.csv)")
uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

if uploaded_file is not None:
    # Đọc file người dùng tải lên
    df_input = pd.read_csv(uploaded_file)
    st.write("Xem trước dữ liệu đầu vào:")
    st.dataframe(df_input.head())

    # Nút bấm dự đoán
    if st.button("Chạy Dự Đoán", type="primary"):
        with st.spinner('AI đang phân tích dữ liệu...'):
            try:
                # Tiền xử lý dữ liệu
                X_test = feature_engineering(df_input)
                
                # Chạy dự đoán
                preds = model.predict(X_test)
                
                # Tạo từ điển map nhãn cho dễ nhìn trên web
                status_mapping = {0: '0 - Normal', 1: '1 - Warning', 2: '2 - Dropout'}
                
                # Tạo bảng kết quả
                result_df = pd.DataFrame({
                    'Student_ID': df_input['Student_ID'] if 'Student_ID' in df_input.columns else [f"SV_Unknown_{i}" for i in range(len(preds))],
                    'Academic_Status': preds,
                    'Label': [status_mapping[p] for p in preds]
                })
                
                st.success("Dự đoán thành công!")
                st.write("Bảng kết quả:")
                st.dataframe(result_df)
                
                # Xuất ra định dạng Kaggle Submission (Chỉ lấy Student_ID và Academic_Status)
                kaggle_submission = result_df[['Student_ID', 'Academic_Status']]
                csv_data = kaggle_submission.to_csv(index=False).encode('utf-8')
                
                # Nút tải file
                st.download_button(
                    label="📥 Tải file nộp bài (submission.csv)",
                    data=csv_data,
                    file_name='submission.csv',
                    mime='text/csv',
                )
                
            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
