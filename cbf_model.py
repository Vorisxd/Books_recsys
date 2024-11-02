import torch
import torch.nn as nn
import pickle
import pandas as pd

with open('bk_ind.pkl', 'rb') as f:
    index_dicts = pickle.load(f)

# Доступ к отдельным словарям
category_to_index = index_dicts['category_to_index']
author_to_index = index_dicts['author_to_index']
title_to_index = index_dicts['title_to_index']

unique_categories_len, unique_authors_len, unique_titles_len = (7424, 107386, 156144)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

title_sentiment_aggregated = pd.read_csv('title_sentiment_aggregateds.csv', keep_default_na=False)

class ContentBasedFilteringModel(nn.Module):
    def __init__(self, num_categories, num_authors, num_titles, embedding_dim):
        super(ContentBasedFilteringModel, self).__init__()
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        self.author_embedding = nn.Embedding(num_authors, embedding_dim)
        self.title_embedding = nn.Embedding(num_titles, embedding_dim)
        self.sentiment_linear = nn.Linear(4 * embedding_dim, 1)

    def forward(self, category_indices, author_indices, title_indices, sentiment_scores):
        category_embedded = self.category_embedding(category_indices)
        author_embedded = self.author_embedding(author_indices)
        title_embedded = self.title_embedding(title_indices)
        sentiment_expanded = sentiment_scores.unsqueeze(1).expand_as(category_embedded)



        concatenated = torch.cat([category_embedded, author_embedded, title_embedded, sentiment_expanded], dim=1)
        output = self.sentiment_linear(concatenated)
        return output
    
num_categories = unique_categories_len
num_authors = unique_authors_len
num_titles = unique_titles_len
embedding_dim = 64  
cbf_model = ContentBasedFilteringModel(num_categories, num_authors, num_titles, embedding_dim)


def func1():
    print(len(title_sentiment_aggregated['Title']))

def get_content_based_recommendations(content_based_model, collaborative_recommendations):
    print(len(title_sentiment_aggregated['Title'].unique()))

    title_details = title_sentiment_aggregated.set_index('Title')[['categories', 'authors', 'sentiment_score']].to_dict(orient='index')

    details = [title_details[title] for title in collaborative_recommendations]


    category_indices = torch.tensor([category_to_index[detail['categories']] for detail in details], dtype=torch.long)
    author_indices = torch.tensor([author_to_index[detail['authors']] for detail in details], dtype=torch.long)
    title_indices = torch.tensor([title_to_index[title] for title in collaborative_recommendations], dtype=torch.long)
    sentiment_scores = torch.tensor([detail['sentiment_score'] for detail in details], dtype=torch.float32)
    category_indices, author_indices, title_indices, sentiment_scores= category_indices.to(device), author_indices.to(device), title_indices.to(device), sentiment_scores.to(device)

    content_based_model.eval()
    with torch.inference_mode():
        predictions = content_based_model(category_indices, author_indices, title_indices, sentiment_scores)


    sorted_titles = [title for _, title in sorted(zip(predictions, collaborative_recommendations), reverse=True)]


    return sorted_titles[100]
