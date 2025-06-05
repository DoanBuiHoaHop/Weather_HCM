import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import os

# Cấu hình trang Streamlit
st.set_page_config(page_title="Phân tích & Dự đoán Thời tiết", layout="wide")

# Tiêu đề ứng dụng
st.title("Ứng dụng Phân tích và Dự đoán Thời tiết 🌦️")

# Sidebar: Cấu hình chung
st.sidebar.header("Cấu hình")
theme = st.sidebar.selectbox("Chọn giao diện", ["Light", "Dark"])
if theme == "Dark":
    sns.set_theme(style="darkgrid")
else:
    sns.set_theme(style="whitegrid")

# Tạo các tab
tab1, tab2, tab3, tab4 = st.tabs(["Tổng quan Dữ liệu", "Phân tích Dữ liệu", "Dự đoán Nhiệt độ", "Dự đoán Tương Lai"])

# Hàm tiền xử lý dữ liệu
def preprocess_data(df, features=None):
    df = df.copy()
    # Ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Drop rows with invalid dates
    df = df.dropna(subset=['Date', 'Temp'])
    
    # Extract useful features from Date before dropping it
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # Tạo đặc trưng trễ
    def derive_nth_day_feature(df, feature, N):
        rows = df.shape[0]
        nth_prior_measurements = [None] * N + [df[feature][i - N] for i in range(N, rows)]
        col_name = f"{feature}_{N}"
        df[col_name] = nth_prior_measurements
    
    if features:
        for feature in features:
            if feature in df.columns and feature != 'Temp':
                # Ensure the feature is numeric before creating lagged features
                if pd.api.types.is_numeric_dtype(df[feature]):
                    for N in range(1, 4):
                        derive_nth_day_feature(df, feature, N)
    
    # Drop the 'Date' column to avoid issues with non-numeric data
    df = df.drop(columns=['Date'], errors='ignore')
    
    # Xóa các dòng chứa giá trị None
    df = df.dropna()
    return df

# Tab 1: Tổng quan Dữ liệu
with tab1:
    st.header("Tổng quan Dữ liệu")
    
    uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"], key="overview")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, sep=None, engine='python')
            
            # Kiểm tra cột bắt buộc
            if 'Date' not in data.columns or 'Temp' not in data.columns:
                st.error("File CSV phải chứa cột 'Date' và 'Temp'.")
                st.stop()
            
            st.write("**Thông tin dữ liệu:**")
            st.write(data.info())
            st.write("**5 dòng đầu tiên:**")
            st.write(data.head())
            
            # Thống kê mô tả
            st.write("**Thống kê mô tả:**")
            st.write(data.describe())
            
            # Hiển thị số lượng giá trị thiếu
            st.write("**Giá trị thiếu:**")
            st.write(data.isnull().sum())
            
        except Exception as e:
            st.error(f"Lỗi khi xử lý dữ liệu: {e}")
    else:
        st.info("Vui lòng tải lên file CSV để xem tổng quan dữ liệu.")

# Tab 2: Phân tích Dữ liệu
with tab2:
    st.header("Phân tích Dữ liệu Thời tiết")
    
    uploaded_file_analysis = st.file_uploader("Tải lên file CSV", type=["csv"], key="analysis")
    
    if uploaded_file_analysis is not None:
        try:
            data = pd.read_csv(uploaded_file_analysis, sep=None, engine='python')
            
            if 'Date' not in data.columns or 'Temp' not in data.columns:
                st.error("File CSV phải chứa cột 'Date' và 'Temp'.")
                st.stop()
            
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.dropna(subset=['Date'])
            
            # Bộ lọc thời gian
            min_date = data['Date'].min().date()
            max_date = data['Date'].max().date()
            start_date, end_date = st.date_input(
                "Chọn khoảng thời gian",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
            
            mask = (data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))
            data = data.loc[mask]
            
            data['Month'] = data['Date'].dt.month
            data['Year'] = data['Date'].dt.year
            
            # Lựa chọn đặc trưng để phân tích
            available_features = [col for col in data.columns if col not in ['Date', 'Time']]
            selected_feature = st.selectbox("Chọn đặc trưng để phân tích", available_features, index=available_features.index('Temp'))
            
            # Phân tích theo tháng
            if selected_feature:
                mean_by_month = data.groupby('Month')[selected_feature].mean()
                
                st.subheader(f"{selected_feature} trung bình theo tháng")
                chart_data = {
                    "type": "bar",
                    "data": {
                        "labels": [calendar.month_abbr[i] for i in range(1, 13)],
                        "datasets": [{
                            "label": f"{selected_feature}",
                            "data": mean_by_month.tolist(),
                            "backgroundColor": "rgba(70, 130, 180, 0.7)",
                            "borderColor": "rgba(70, 130, 180, 1)",
                            "borderWidth": 1
                        }]
                    },
                    "options": {
                        "scales": {
                            "y": {
                                "beginAtZero": False,
                                "title": {"display": True, "text": selected_feature}
                            },
                            "x": {
                                "title": {"display": True, "text": "Tháng"}
                            }
                        },
                        "plugins": {
                            "title": {
                                "display": True,
                                "text": f"{selected_feature} trung bình theo tháng tại TP.HCM"
                            }
                        }
                    }
                }
                st.write("```chartjs\n" + str(chart_data) + "\n```")
                
                # Hiển thị giá trị cao nhất và thấp nhất
                max_month = mean_by_month.idxmax()
                min_month = mean_by_month.idxmin()
                st.write(f"Giá trị cao nhất: {mean_by_month.max():.1f} (Tháng {calendar.month_abbr[max_month]})")
                st.write(f"Giá trị thấp nhất: {mean_by_month.min():.1f} (Tháng {calendar.month_abbr[min_month]})")
            
            # Phân tích xu hướng theo năm
            if st.checkbox("Hiển thị xu hướng theo năm"):
                mean_by_year = data.groupby('Year')[selected_feature].mean()
                st.subheader(f"{selected_feature} trung bình theo năm")
                chart_year = {
                    "type": "line",
                    "data": {
                        "labels": mean_by_year.index.tolist(),
                        "datasets": [{
                            "label": f"{selected_feature}",
                            "data": mean_by_year.tolist(),
                            "borderColor": "rgba(220, 20, 60, 1)",
                            "backgroundColor": "rgba(220, 20, 60, 0.2)",
                            "fill": True,
                            "tension": 0.4
                        }]
                    },
                    "options": {
                        "scales": {
                            "y": {
                                "beginAtZero": False,
                                "title": {"display": True, "text": selected_feature}
                            },
                            "x": {
                                "title": {"display": True, "text": "Năm"}
                            }
                        },
                        "plugins": {
                            "title": {
                                "display": True,
                                "text": f"{selected_feature} trung bình theo năm tại TP.HCM"
                            }
                        }
                    }
                }
                st.write("```chartjs\n" + str(chart_year) + "\n```")
                
        except Exception as e:
            st.error(f"Lỗi khi xử lý dữ liệu: {e}")
    else:
        st.info("Vui lòng tải lên file CSV để phân tích.")

# Tab 3: Dự đoán Nhiệt độ
with tab3:
    st.header("Dự đoán Nhiệt độ")
    
    uploaded_file_pred = st.file_uploader("Tải lên file CSV", type=["csv"], key="prediction")
    
    if uploaded_file_pred is not None:
        try:
            data_temp = pd.read_csv(uploaded_file_pred, sep='\t')
            
            # Hiển thị thông tin dữ liệu
            st.write("**Thông tin dữ liệu:**")
            st.write(data_temp.info())
            st.write("**5 dòng đầu tiên:**")
            st.write(data_temp.head())
            
            # Lựa chọn đặc trưng
            available_features = [col for col in data_temp.columns if col not in ['Time', 'Date', 'Temp']]
            selected_features = st.multiselect("Chọn đặc trưng để huấn luyện", available_features, default=available_features[:min(3, len(available_features))])
            
            # Tiền xử lý dữ liệu
            data_temp = preprocess_data(data_temp, selected_features)
            
            # Lọc các cột có tương quan thấp
            corr = data_temp.select_dtypes(include='number').corr()[['Temp']].sort_values('Temp')
            data_temp.drop(columns=[col for col in corr.index if abs(corr['Temp'].loc[col]) < 0.5 and col != 'Temp'], inplace=True)
            
            # Mã hóa one-hot encoding
            cols_object = [col for col in data_temp.columns if data_temp[col].dtype == 'object']
            for col in cols_object:
                dummies = pd.get_dummies(data_temp[col], prefix=col)
                data_temp.drop(columns=[col], inplace=True)
                data_temp = pd.concat([data_temp, dummies], axis=1)
            
            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X = scaler.fit_transform(data_temp.drop(columns=['Temp']).values)
            y = data_temp['Temp'].values
            
            # Chia dữ liệu
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Lựa chọn mô hình
            model_options = ['Linear Regression', 'Polynomial Regression', 'Random Forest', 'XGBoost']
            selected_models = st.multiselect("Chọn mô hình", model_options, default=['Linear Regression', 'Random Forest'])
            
            # Tham số mô hình
            model_params = {}
            if 'Polynomial Regression' in selected_models:
                degree = st.slider("Chọn bậc của Polynomial Regression", 1, 5, 2)
                model_params['Polynomial Regression'] = {'degree': degree}
            if 'Random Forest' in selected_models:
                n_estimators = st.slider("Số cây trong Random Forest", 50, 200, 100)
                model_params['Random Forest'] = {'n_estimators': n_estimators}
            if 'XGBoost' in selected_models:
                learning_rate = st.slider("Learning rate của XGBoost", 0.01, 0.3, 0.1)
                model_params['XGBoost'] = {'learning_rate': learning_rate}
            
            # Khởi tạo mô hình
            models = {
                'Linear Regression': LinearRegression(),
                'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=model_params.get('Polynomial Regression', {}).get('degree', 2)), LinearRegression()),
                'Random Forest': RandomForestRegressor(n_estimators=model_params.get('Random Forest', {}).get('n_estimators', 100), random_state=42),
                'XGBoost': xgb.XGBRegressor(learning_rate=model_params.get('XGBoost', {}).get('learning_rate', 0.1), random_state=42)
            }
            
            # Huấn luyện và đánh giá
            results = {}
            for name in selected_models:
                model = models[name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = {
                    'y_pred': y_pred,
                    'r2': r2_score(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': mean_squared_error(y_test, y_pred, squared=False)
                }
            
            # Hiển thị kết quả
            st.subheader("Kết quả mô hình")
            for name, result in results.items():
                st.write(f"**{name}:** R² = {result['r2']:.2f}, MAE = {result['mae']:.2f}, RMSE = {result['rmse']:.2f}")
            
            # Lưu mô hình
            if st.button("Lưu mô hình tốt nhất"):
                best_model_name = max(results, key=lambda x: results[x]['r2'])
                joblib.dump(models[best_model_name], f"{best_model_name}_model.pkl")
                joblib.dump(scaler, "scaler.pkl")
                st.success(f"Đã lưu mô hình {best_model_name} và scaler")
            
            # Vẽ biểu đồ so sánh
            plt.figure(figsize=(12, 8))
            plt.scatter(range(len(y_test)), y_test, color='dodgerblue', s=50, label='Dữ liệu gốc', alpha=0.7)
            
            colors = {
                'Linear Regression': 'crimson',
                'Polynomial Regression': 'limegreen',
                'Random Forest': 'purple',
                'XGBoost': 'orange'
            }
            for name, result in results.items():
                plt.plot(range(len(y_test)), result['y_pred'], color=colors[name], linewidth=2.5, label=name)
            
            plt.title('So sánh các mô hình dự đoán', fontsize=18, fontweight='bold')
            plt.xlabel('Chỉ số', fontsize=14)
            plt.ylabel('Nhiệt độ (°C)', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(plt)
            
        except Exception as e:
            st.error(f"Lỗi khi xử lý dữ liệu: {e}")
    else:
        st.info("Vui lòng tải lên file CSV để dự đoán.")