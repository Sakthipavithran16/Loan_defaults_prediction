import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import datetime
import pickle
from product_recomend_functions import create_interaction_matrix, calculate_item_similarity, get_item_based_recommendations


df = pd.read_csv(r"D:/Banking_Final_Project/Final_loan_df.csv")

product_df = pd.read_csv(r"D:/Banking_Final_Project/product_data.csv")


# Stramlit UI
st.set_page_config(layout="wide")

st.title("Banking Predictions")

option = option_menu(None,options = ["Loan Default Prediction","Customer Segmentation","Product Recommendation"],
                       icons = ["bank","circle-half","cart-check"],
                       default_index=0,
                       orientation="horizontal", 
                       styles={"nav-link-selected": {"background-color": "#3dfc0a"}})


Term_options  = list(df['Term'].unique())
Purpose_options  = list(df['Purpose'].unique())

Type_options  = ['deposit','withdrawal','Transfers','Loan and credit','Bill Payments']
min_date = datetime.date(2023, 9, 23)
max_date = datetime.date(2024, 9, 23)


if option == "Loan Default Prediction":

    with st.form('Classification'):

        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            st.write(' ')
            Loan_Amount = st.text_input('Loan Amount (Min: 11242, Max: 789250)')
            Credit_Score = st.text_input('Credit Score (Min: 585, Max: 7510)')
            Annual_Income =st.text_input('Annual Income (Min: 76627, Max: 165557393)')

        with col3:
            st.write(' ')        
            No_of_Accounts =st.text_input('No of Accounts (Min: 1, Max: 76)')
            Term =st.selectbox('Term', sorted(Term_options))
            Purpose = st.selectbox('Purpose', sorted(Purpose_options))
            submit_button_loan  = st.form_submit_button(label='SUBMIT')


    if submit_button_loan :

        with open('loan_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)

        with open('loan_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('loan_defaults_model.pkl', 'rb') as f:
            loan_model = pickle.load(f)

        new_data = np.array([[int(Loan_Amount), np.log(float(Credit_Score)), np.log(float(Annual_Income)),
                              np.log(float(No_of_Accounts)), Term, Purpose]])
        
        new_data_encoded = encoder.transform(new_data[:, 4::])
        new_data = np.concatenate((new_data[:, [0, 1, 2, 3]],new_data_encoded), axis=1)


        new_data = scaler.transform(new_data)
        new_pred = loan_model.predict(new_data)

        if new_pred == 1:
            st.write('## :red[Loan Default] ')
        else:
            st.write('## :green[Non-Default] ')


elif option == "Customer Segmentation":

    with st.form('Customer_Segmentation'):   

        st.write(' ')
        transaction_amount = st.text_input('Transaction Amount (Min: 10, Max: 500)')
        transaction_type = st.selectbox('Transaction Type', sorted(Type_options))
        transaction_date = st.date_input("Transaction date",None,
                            min_value=min_date, 
                            max_value=max_date)

        submit_button_seg  = st.form_submit_button(label='SUBMIT')


        if submit_button_seg :

            with open('cus_seg_encoder.pkl', 'rb') as f:
                encoder_seg = pickle.load(f)

            with open('cus_seg__scaler.pkl', 'rb') as f:
                scaler_seg = pickle.load(f)

            with open('kmeans_model.pkl', 'rb') as f:
                cus_seg_model = pickle.load(f)

            year = int(transaction_date.year)
            month = int(transaction_date.month)
            day = int(transaction_date.day)

            new_cust_data = np.array([[float(transaction_amount),transaction_type,year,month,day]])
            cust_data_encoded = encoder_seg.transform(new_cust_data[:, [1]])
            new_cust_data = np.concatenate((new_cust_data[:, [0, 2, 3,4]],cust_data_encoded), axis=1)

            new_cust_data = scaler_seg.transform(new_cust_data)
            predicted_segment = cus_seg_model.predict(new_cust_data)

            st.write(f' ## :green[The predicted customer segment is: {predicted_segment[0]}]')


elif option == "Product Recommendation":

    customer_id  = st.text_input('Customer ID  (Min: CID0001, Max: CID1000)')

    st.write(' ')

    sub_button = st.button('Submit')

    if sub_button:

        # Create the customer-product interaction matrix
        interaction_matrix = create_interaction_matrix(product_df)

        # Calculate item-item similarity
        item_similarity_df = calculate_item_similarity(interaction_matrix)

        # Get top 5 product recommendations for the customer
        recommended_products = get_item_based_recommendations(customer_id, interaction_matrix, item_similarity_df, n=5)

        st.write(f"## :green[Recommended products for {customer_id}:] {', '.join(recommended_products)}")
