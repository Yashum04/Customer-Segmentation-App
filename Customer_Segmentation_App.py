import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('customer_kmeans.pkl','rb'))
scaler = pickle.load(open('customer_scaler.pkl','rb'))

st.title('📊 Customer Segmentation')

income = st.number_input('Annual Income', 1000, 200000, 50000)
purchase = st.number_input('Purchase Amount', 0, 100000, 2000)
frequency = st.number_input('Purchase Frequency', 1, 50, 5)
loyalty = st.number_input('Loyalty Score', 1, 100, 50)

if st.button('Predict Segment'):
    data = np.array([[income, purchase, frequency, loyalty]])
    data_scaled = scaler.transform(data)

    cluster = model.predict(data_scaled)[0]

    st.success(f'Customer belongs to Segment {cluster}')

    if cluster == 0:
        st.info('💎 High Value Customer')
    elif cluster == 1:
        st.warning('⚠️ Medium Value Customer')
    else:
        st.error('❌ Low Value Customer')