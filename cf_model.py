import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F



with open('index_dicts.pkl', 'rb') as f:
    index_dicts = pickle.load(f)

# Извлечение словарей из загруженного объекта
user_to_index = index_dicts['user_to_index']
item_to_index = index_dicts['item_to_index']
index_to_title = index_dicts['index_to_title']

class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.hidden_layer = nn.Linear(embedding_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, user_indices, item_indices):
        user_embedded = self.user_embedding(user_indices)
        item_embedded = self.item_embedding(item_indices)
        concatenated = torch.cat([user_embedded, item_embedded], dim=1)
        hidden_output = self.relu(self.hidden_layer(concatenated))
        output = self.output_layer(hidden_output)
        return output
    
    def get_similar_titles(self, input_title_index, top_k=100):
        device = self.item_embedding.weight.device  # Get the device of the embeddings

        # Move the input title index to the same device as the model
        input_title_index = torch.tensor([input_title_index], device=device)

        # Get the embedding for the input title
        input_title_embedding = self.item_embedding(input_title_index)

        # Get embeddings for all titles
        all_title_embeddings = self.item_embedding.weight

        # Calculate cosine similarity
        similarities = F.cosine_similarity(input_title_embedding, all_title_embeddings)

        # Get indices of top-k similar titles
        #argsort returns the indices that sort a tensor along a given dimension in ascending order(default) by value.
        similar_title_indices = torch.argsort(similarities, descending=True)[:top_k] 
        
        # Convert indices to a list of titles
        similar_titles = [index_to_title[idx.item()] for idx in similar_title_indices]
        #we are using item() to get scalar value instead of tensor which can be used as an key index for dictionary
        return similar_titles


num_users = 876268
num_items = 156144
embedding_dim = 100  # You can adjust this dimension based on your needs
model = CollaborativeFilteringModel(num_users, num_items, embedding_dim, 32)

def get_collaborative_recommendations(model, title, num_recommendations=100):
    
    
    #item_to_index = {title: idx for idx, title in enumerate(item_ids)}
    # Get index of the input title
    input_title_index = item_to_index[title] # have already declared this before in above cells

    # Get recommendations using the collaborative filtering model
    model.eval()
    with torch.inference_mode():
        # Call the custom method to get similar titles
        similar_titles = model.get_similar_titles(input_title_index, top_k=num_recommendations)

    
    # Return the recommended titles
    return similar_titles


































# # Define the content-based filtering model
# class ContentBasedFilteringModel(nn.Module):
#     def __init__(self, num_categories, num_authors, num_titles, embedding_dim):
#         super(ContentBasedFilteringModel, self).__init__()
#         self.category_embedding = nn.Embedding(num_categories, embedding_dim)
#         self.author_embedding = nn.Embedding(num_authors, embedding_dim)
#         self.title_embedding = nn.Embedding(num_titles, embedding_dim)
#         self.sentiment_linear = nn.Linear(4 * embedding_dim, 1)

#     def forward(self, category_indices, author_indices, title_indices, sentiment_scores):
#         category_embedded = self.category_embedding(category_indices)
#         author_embedded = self.author_embedding(author_indices)
#         title_embedded = self.title_embedding(title_indices)
#         sentiment_expanded = sentiment_scores.unsqueeze(1).expand_as(category_embedded) 
#         # It serves as a constant tensor that gets expanded to match the size of category_embedded for concatenation, and its values remain fixed throughout training
#         #self.expand_as(other) is equivalent to self.expand(other.size()).

#         concatenated = torch.cat([category_embedded, author_embedded, title_embedded, sentiment_expanded], dim=1)
#         output = self.sentiment_linear(concatenated)
#         return output
    
   