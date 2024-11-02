import pickle
import streamlit as st
import torch
from cf_model import CollaborativeFilteringModel, get_collaborative_recommendations
from cbf_model import ContentBasedFilteringModel, get_content_based_recommendations

# Загрузка моделей
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfm_model_path = r"pre_trained_models\collaborative_filtering_model8.pth"
cfm_model = torch.load(cfm_model_path, map_location=device, weights_only=False)
cfm_model.to(device)

cbf_model_path = r"pre_trained_models\content_based_filtering_model4.pth"
cbf_model = torch.load(cbf_model_path, map_location=device, weights_only=False)
cbf_model.to(device)

# Загрузка словаря с названиями книг и URL их обложек
with open('title_image.pkl', 'rb') as f:
    index_dicts = pickle.load(f)
title_image_dict = index_dicts['title_image_dict']

# Функция для поиска похожих названий
def search_similar_titles(input_text, title_dict, max_results=10):
    input_text_lower = input_text.lower()
    similar_titles = [title for title in title_dict.keys() if input_text_lower in title.lower()]
    return similar_titles[:max_results]

# Заголовок приложения
st.title("Список книг и их обложки")

# Поле ввода для названия книги
input_title = st.text_input("Введите название книги", "")

# Обновление списка предложений при вводе
suggestions = search_similar_titles(input_title, title_image_dict) if input_title else []

# Вывод предложений как кнопок
selected_title = None
for title in suggestions:
    if st.button(title):
        selected_title = title
        break

# Если выбрано название книги, отображаем рекомендации
if selected_title:
    collab_recommendations = get_collaborative_recommendations(cfm_model, selected_title, num_recommendations=1000)
    sorted_recs = get_content_based_recommendations(cbf_model, collab_recommendations)
    filtered_dict = {key:title_image_dict[key] for key in collab_recommendations }


    # Вывод первых 40 книг в виде таблицы (4 колонки на 10 строк)
    num_books = min(40, len(filtered_dict))  # Ограничиваем до 40 книг
    columns = st.columns(4)  # Создаем четыре колонки

    for i, (title, image_url) in enumerate(list(filtered_dict.items())[:num_books]):
        col_index = i % 4  # Индекс колонки
        with columns[col_index]:
            st.subheader(title)  # Название книги
            st.image(image_url, use_column_width=True)  # Обложка книги
