import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

pipeline = pickle.load(open('model_pipeline.pkl', 'rb'))

df = pd.read_excel('Telco_customer_churn.xlsx')

num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Value', 'Churn Score', 'CLTV']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=num_cols)

st.sidebar.markdown(
    """
    <style>
    .css-1d391kg {
        background-color: #5b9bd5; /* Sidebar Blue */
    }
    .css-1d391kg a {
        color: white;
    }
    .css-1d391kg .stSidebar > div:first-child {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title('Customer Churn Dashboard')
menu = st.sidebar.radio('Menu', ['Home', 'Predict Churn'])

if menu == 'Home':
    st.markdown("<h1 style='text-align: center; color: teal;'>Customer Churn Dashboard</h1>", unsafe_allow_html=True)
    st.write("")

    st.image('Customer_Churn.png', use_container_width=True)
    st.write("---")

    st.subheader('Distribusi Churn Pelanggan', anchor='dist-churn')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Churn Label', hue='Churn Label', data=df, palette='Set2', legend=False, ax=ax)
    ax.set_xlabel('Churn (Yes / No)')
    ax.set_ylabel('Jumlah Pelanggan')
    st.pyplot(fig)

    st.subheader('Heatmap Korelasi Fitur Numerik')
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.subheader('Boxplot Monthly Charges vs Churn')
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Churn Label', y='Monthly Charges', data=df, palette='Set1', ax=ax3)
    ax3.set_xlabel('Churn')
    ax3.set_ylabel('Monthly Charges')
    st.pyplot(fig3)

    st.subheader('Rata-rata Tenure Berdasarkan Churn')
    avg_tenure = df.groupby('Churn Label')['Tenure Months'].mean().reset_index()
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.barplot(x='Churn Label', y='Tenure Months', data=avg_tenure, palette='pastel', ax=ax4)
    ax4.set_ylabel('Rata-rata Tenure (bulan)')
    st.pyplot(fig4)

    st.subheader('Distribusi Total Charges Berdasarkan Churn')
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x='Total Charges', hue='Churn Label', bins=30, kde=True, palette='Set2', ax=ax5)
    ax5.set_xlim(0, df['Total Charges'].quantile(0.95))  # fokus ke 95% data
    st.pyplot(fig5)


elif menu == 'Predict Churn':
    st.markdown("<h1 style='text-align: center; color: teal;'>Prediksi Pelanggan Churn</h1>", unsafe_allow_html=True)
    st.write("---")

    gender = st.selectbox('Gender', ['Female', 'Male'])
    senior = st.selectbox('Senior Citizen', ['No', 'Yes'])
    tenure_months = st.number_input('Tenure Months', min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=50.0)
    total_charges = st.number_input('Total Charges', min_value=0.0, value=monthly_charges * tenure_months)

    df_input = pd.DataFrame({
        'Gender': [0 if gender == 'Female' else 1],
        'Senior Citizen': [0 if senior == 'No' else 1],
        'Tenure Months': [tenure_months],
        'Monthly Charges': [monthly_charges],
        'Total Charges': [total_charges]
    })

    if st.button('Predict'):
        pred = pipeline.predict(df_input)[0]
        if pred == 1:
            st.error('🚨 Pelanggan kemungkinan CHURN')
        else:
            st.success('✅ Pelanggan kemungkinan LOYAL')
