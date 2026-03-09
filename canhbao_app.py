import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

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
    if pd.isna(x): return 'unknown'
    return str(x).lower().strip()

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Dự đoán Trạng thái Học tập", layout="wide")
st.title("🎓 Ứng dụng Dự đoán Trạng thái Học tập Sinh viên")
st.markdown("Tải lên file `train.csv` và `test.csv` từ máy của bạn. Hệ thống sẽ tự động huấn luyện mô hình và xuất ra file kết quả để nộp lên Kaggle.")

# Khu vực tải file
col1, col2 = st.columns(2)
with col1:
    train_file = st.file_uploader("1. Tải lên tập huấn luyện (train.csv)", type=["csv"])
with col2:
    test_file = st.file_uploader("2. Tải lên tập kiểm tra (test.csv)", type=["csv"])

if train_file is not None and test_file is not None:
    st.success("Đã tải lên đủ 2 file! Nhấn nút bên dưới để bắt đầu chạy mô hình.")
    
    if st.button("🚀 Bắt đầu Huấn luyện & Dự đoán", use_container_width=True):
        with st.spinner("Đang đọc và xử lý dữ liệu... Vui lòng đợi..."):
            # 1. ĐỌC DỮ LIỆU
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            
            target_col = 'Academic_Status'
            y_train = train_df[target_col].values
            test_ids = test_df['Student_ID']
            
            all_df = pd.concat([train_df.drop(columns=[target_col]), test_df], axis=0).reset_index(drop=True)
            
            # 2. LÀM SẠCH DỮ LIỆU
            all_df['English_Level'] = all_df['English_Level'].apply(clean_english_level)
            all_df['Admission_Mode'] = all_df['Admission_Mode'].apply(clean_admission_mode)
            
            num_cols = ['Age', 'Tuition_Debt', 'Count_F', 'Training_Score_Mixed'] + [col for col in all_df.columns if 'Att_Subject' in col]
            for col in num_cols:
                all_df[col] = pd.to_numeric(all_df[col], errors='coerce').fillna(-1)
                
            cat_cols = ['Gender', 'Hometown', 'Current_Address', 'Admission_Mode', 'English_Level', 'Club_Member']
            for col in cat_cols:
                all_df[col] = all_df[col].astype(str).fillna('unknown')
                le = LabelEncoder()
                all_df[col] = le.fit_transform(all_df[col])
                
            all_df['Advisor_Notes'] = all_df['Advisor_Notes'].apply(clean_text)
            all_df['Personal_Essay'] = all_df['Personal_Essay'].apply(clean_text)
            all_df['Combined_Text'] = all_df['Advisor_Notes'] + " " + all_df['Personal_Essay']
            
            # 3. FEATURE ENGINEERING (TF-IDF)
            st.info("Đang trích xuất đặc trưng văn bản bằng TF-IDF...")
            tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            text_features = tfidf.fit_transform(all_df['Combined_Text'])
            
            X_tabular = all_df.drop(columns=['Student_ID', 'Advisor_Notes', 'Personal_Essay', 'Combined_Text'])
            
            X_train_sparse = csr_matrix(X_tabular.iloc[:len(train_df)].values)
            X_test_sparse = csr_matrix(X_tabular.iloc[len(train_df):].values)
            
            text_train = text_features[:len(train_df)]
            text_test = text_features[len(train_df):]
            
            X_train_final = hstack([X_train_sparse, text_train]).tocsr()
            X_test_final = hstack([X_test_sparse, text_test]).tocsr()
            
            # 4. HUẤN LUYỆN MÔ HÌNH
            st.info("Đang huấn luyện mô hình LightGBM (5-Fold CV)...")
            N_SPLITS = 5
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
            
            oof_preds = np.zeros((X_train_final.shape[0], 3))
            test_preds = np.zeros((X_test_final.shape[0], 3))
            
            lgb_params = {
                'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
                'boosting_type': 'gbdt', 'learning_rate': 0.05, 'num_leaves': 31,
                'feature_fraction': 0.8, 'class_weight': 'balanced',
                'random_state': 42, 'n_jobs': -1, 'verbose': -1
            }
            
            progress_bar = st.progress(0)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_final, y_train)):
                X_tr, y_tr = X_train_final[train_idx], y_train[train_idx]
                X_va, y_va = X_train_final[val_idx], y_train[val_idx]
                
                train_data = lgb.Dataset(X_tr, label=y_tr)
                val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
                
                model = lgb.train(
                    lgb_params, train_data, num_boost_round=1000,
                    valid_sets=[train_data, val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                
                oof_preds[val_idx] = model.predict(X_va)
                test_preds += model.predict(X_test_final) / N_SPLITS
                
                # Cập nhật thanh tiến trình
                progress_bar.progress((fold + 1) / N_SPLITS)
            
            # 5. TỔNG KẾT & XUẤT KẾT QUẢ
            oof_pred_classes = np.argmax(oof_preds, axis=1)
            cv_macro_f1 = f1_score(y_train, oof_pred_classes, average='macro')
            
            st.success(f"✅ Hoàn tất huấn luyện! Điểm CV Macro F1: **{cv_macro_f1:.4f}**")
            
            final_test_preds = np.argmax(test_preds, axis=1)
            submission = pd.DataFrame({
                'Student_ID': test_ids,
                'Academic_Status': final_test_preds
            })
            
            # Chuyển DataFrame thành dạng CSV trong bộ nhớ để tải xuống
            csv_data = submission.to_csv(index=False).encode('utf-8')
            
            st.markdown("### 📥 Tải kết quả")
            st.download_button(
                label="Tải file submission.csv",
                data=csv_data,
                file_name='submission.csv',
                mime='text/csv',
                type="primary"
            )
else:
    st.warning("Vui lòng tải lên cả 2 file `train.csv` và `test.csv` để tiếp tục.")
