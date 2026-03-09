import streamlit as st 
import pandas as pd
import pickle
@st.cache_resource
def load_model():
    with open('student_dropout_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.subheader("Tải lên dữ liệu sinh viên cần dự đoán (File CSV)")
uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.dataframe(df_input.head()) # Hiển thị 5 dòng đầu tiên lên web cho đẹp

if st.button("Dự đoán"): # Khi người dùng bấm nút
        # 1. Tiền xử lý file test giống hệt lúc train
        X_test = feature_engineering(df_input)
        
        # 2. Đưa qua mô hình để dự đoán (ra mảng các số 0, 1, 2)
        preds = model.predict(X_test)
        
        # 3. Tạo bảng kết quả mới có 2 cột đúng format Kaggle yêu cầu
        result_df = pd.DataFrame({
            'Student_ID': df_input['Student_ID'],
            'Academic_Status': preds
        })
        
        # 4. Chuyển bảng kết quả này thành dạng văn bản CSV
        csv = result_df.to_csv(index=False).encode('utf-8')
        
        # 5. Tạo nút Download để người dùng tải file submission.csv về máy
        st.download_button(
            label="📥 Tải file Submission (CSV)",
            data=csv,
            file_name='submission.csv',
            mime='text/csv',
        )
    
