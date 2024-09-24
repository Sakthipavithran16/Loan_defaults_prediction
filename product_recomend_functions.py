import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Create Customer-Product Interaction Matrix
def create_interaction_matrix(df):
    interaction_mapping = {'purchased': 1.0, 'viewed': 0.5, 'clicked': 0.2}
    
    df['interaction_value'] = df['interaction_type'].map(interaction_mapping)
    
    # Pivot the table to create a customer-product interaction matrix
    interaction_matrix = df.pivot_table(index='customer_id', columns='product_id', values='interaction_value').fillna(0)
    
    return interaction_matrix


# Calculate Item-Item Similarity (Cosine Similarity between products)
def calculate_item_similarity(interaction_matrix):
    interaction_matrix_np = interaction_matrix.values
    item_similarity = cosine_similarity(interaction_matrix_np.T)  # Transpose to get item-item similarity
    item_similarity_df = pd.DataFrame(item_similarity, index=interaction_matrix.columns, columns=interaction_matrix.columns)
    
    return item_similarity_df

# Step 3: Generate Product Recommendations
def get_item_based_recommendations(customer_id, interaction_matrix, item_similarity_df, n=5):
    customer_interactions = interaction_matrix.loc[customer_id]    # Get the interaction data for the specific customer

    interacted_products = customer_interactions[customer_interactions > 0].index    # Get products the customer has already interacted with
    
    product_scores = pd.Series(np.zeros(interaction_matrix.shape[1]), index=interaction_matrix.columns)  # Initialize an empty score for each product
    
    # Loop through each product the customer has interacted with
    for product in interacted_products:
        similar_products = item_similarity_df[product]    # Get similar products to this product
        
        product_scores += similar_products * customer_interactions[product]    # Add the product similarity to the scores
    
    product_scores = product_scores.drop(interacted_products)   # Remove products the customer has already interacted with
    

    return product_scores.nlargest(n).index    # Return the top N recommended products


