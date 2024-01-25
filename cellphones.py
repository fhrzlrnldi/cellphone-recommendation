from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset cellphones_data and cellphones_ratings
data_cellphones = pd.read_csv('cellphones_data.csv', sep=';')
data_ratings = pd.read_csv('cellphones_ratings.csv', sep=';')

# Merge data based on 'model' column
merged_data = pd.merge(data_cellphones, data_ratings, on='cellphone_id', how='inner')

# Preprocessing text data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()  # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()  # Tokenize text into words
    
    # Remove stopwords and lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)  # Return processed text as a single string

# Preprocess 'clean_date' , 'clean_memory' and 'clean_ram' columns
merged_data['clean_date'] = merged_data['release_date'].apply(preprocess_text)
merged_data['clean_memory'] = merged_data['internal_memory'].apply(preprocess_text)
merged_data['clean_ram'] = merged_data['RAM'].apply(preprocess_text)

# TF-IDF Vectorization for date, memory, and ram
tfidf_date_vectorizer = TfidfVectorizer()
tfidf_memory_vectorizer = TfidfVectorizer()
tfidf_ram_vectorizer = TfidfVectorizer()
tfidf_date_matrix = tfidf_date_vectorizer.fit_transform(merged_data['clean_date'])
tfidf_memory_matrix = tfidf_memory_vectorizer.fit_transform(merged_data['clean_memory'])
tfidf_ram_matrix = tfidf_ram_vectorizer.fit_transform(merged_data['clean_ram'])

# Compute cosine similarity for date, memory, and ram
cosine_sim_date = cosine_similarity(tfidf_date_matrix, tfidf_date_matrix)
cosine_sim_memory = cosine_similarity(tfidf_memory_matrix, tfidf_memory_matrix)
cosine_sim_ram = cosine_similarity(tfidf_ram_matrix, tfidf_ram_matrix)

def get_memory_based_recommendations(entry_text, add_text):
    matching_threshold = 70  # Sesuaikan threshold sesuai kebutuhan
    recommendations_memory = get_recommendations_by_column(entry_text, 'clean_memory')
    recommendations_ram = get_recommendations_by_column(add_text, 'clean_ram')
    return recommendations_memory, recommendations_ram

def get_recommendations_by_column(entry_text_memory, entry_text_ram):
    combined_text = f"{entry_text_memory} {entry_text_ram}"
    filtered_data = merged_data[
        (merged_data['clean_memory'].str.contains(entry_text_memory.lower())) &
        (merged_data['clean_ram'].str.contains(entry_text_ram.lower()))
    ]

    if len(filtered_data) == 0:
        return "No matching cellphones found for the given memory and RAM."
    else:
        # Sorting by rating in descending order
        sorted_data = filtered_data.sort_values(by='rating', ascending=False).head(10)

        recommendations = []
        for index, row in sorted_data.iterrows():
            recommendations.append(f"Cellphone: {row['brand']} {row['model']} (Rating: {row['rating']})")
        return recommendations


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            entry_memory = request.form['entry_memory']
            entry_ram = request.form['entry_ram']
        except KeyError as e:
            return f'Error: Key not found - {e}', 400

        recommendations = get_recommendations_by_column(entry_memory, entry_ram)

        print("Recommendations:", recommendations)
        return render_template('index.html', recommendations=recommendations)   
    return render_template('index.html', recommendations=None)

if __name__ == '__main__':
    app.run(debug=True)
