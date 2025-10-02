import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords if not already present
# FIX: Changed nltk.downloader.DownloadError to LookupError for compatibility
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.info("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# --- CONFIGURATION ---
# Using 'r' prefix for a raw string to correctly handle Windows backslashes
FILE_PATH = r"faq.csv"
SIMILARITY_THRESHOLD = 0.35 # Minimum similarity score to suggest a question
TOP_N_SUGGESTIONS = 3

# --- DATA LOADING AND PREPROCESSING ---
@st.cache_resource
def load_and_prepare_data(file_path):
    """Loads CSV, cleans text, and initializes the TF-IDF vectorizer."""
    try:
        # NOTE: Using a simple read_csv assumes the file is comma-separated
        df = pd.read_csv(file_path)
        
        # Ensure 'Question' and 'Answer' columns exist and drop duplicates/NaT
        # Assuming the CSV has exactly two columns (Q and A)
        if df.shape[1] < 2:
            st.error("Error: CSV must have at least two columns for Question and Answer.")
            st.stop()
            
        df.columns = ['Question', 'Answer'] + list(df.columns[2:])
        df = df[['Question', 'Answer']]
        df.dropna(subset=['Question', 'Answer'], inplace=True)
        df.drop_duplicates(subset=['Question'], inplace=True)
        
        # Preprocessing function
        def preprocess(text):
            if not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
            tokens = text.split()
            # Assuming English stopwords based on previous code.
            tokens = [w for w in tokens if w not in stopwords.words('english') and len(w) > 1]
            return " ".join(tokens)

        # Apply preprocessing to all questions
        df['Processed_Question'] = df['Question'].apply(preprocess)
        
        # Initialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()
        
        # Fit vectorizer to all processed questions
        tfidf_matrix = vectorizer.fit_transform(df['Processed_Question'])
        
        return df, vectorizer, tfidf_matrix
        
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure the path is correct.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data loading or preprocessing: {e}")
        st.stop()
        
# --- CORE SEARCH LOGIC ---
def find_similar_questions(query, df, vectorizer, tfidf_matrix):
    """
    Calculates cosine similarity between the user query and all existing questions.
    Returns the top N matching questions.
    """
    # 1. Preprocess the user query
    processed_query = ' '.join([w for w in query.lower().split() if w not in stopwords.words('english')])
    
    if not processed_query:
        return [], []
    
    # 2. Vectorize the query using the fitted vectorizer
    query_vector = vectorizer.transform([processed_query])
    
    # 3. Calculate cosine similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # 4. Get indices of the top N scores
    
    # Find indices where similarity is above the threshold
    valid_indices = similarity_scores >= SIMILARITY_THRESHOLD
    filtered_scores = similarity_scores[valid_indices]
    
    # Get the original DataFrame indices for the filtered scores
    df_indices = df.index[valid_indices].tolist()
    
    # Combine scores and original indices, then sort
    scored_matches = sorted(
        zip(filtered_scores, df_indices),
        key=lambda x: x[0], 
        reverse=True
    )
    
    # Extract the top N results
    top_matches = scored_matches[:TOP_N_SUGGESTIONS]
    
    # Prepare output
    suggested_questions = []
    suggested_answers = []

    for score, index in top_matches:
        # Check if the score is still useful 
        if score > 0:
            suggested_questions.append(df.loc[index, 'Question'])
            suggested_answers.append(df.loc[index, 'Answer'])
            
    return suggested_questions, suggested_answers


# --- STREAMLIT APP LAYOUT ---
def faq_app():
    st.set_page_config(page_title="FAQ Jadwal Langkah-langkah Akhir Tahun 2025 oleh PPA I DJPb Provinsi Riau", layout="centered")
    
    st.title("🤖 FAQ Jadwal Langkah-langkah Akhir Tahun 2025 oleh PPA I DJPb Provinsi Riau")
    st.markdown("Silahkan masukkan pertanyaan, nanti saya akan tampilkan informasi paling sesuai dari dataset.")

    # Load data only once
    df, vectorizer, tfidf_matrix = load_and_prepare_data(FILE_PATH)
    
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = None
        st.session_state.suggested_q = None

    # --- User Input ---
    user_query = st.text_input("Tanyakan tentang schedule:", key="user_input")
    
    if user_query:
        suggestions, answers = find_similar_questions(user_query, df, vectorizer, tfidf_matrix)
        
        if suggestions:
            st.subheader(f"🤔 Apakah ini maksud Anda? ({len(suggestions)} informasi terkait ditemukan)")
            
            # --- Display Suggestions as Buttons ---
            
            cols = st.columns(len(suggestions))
            
            for i, (q, a) in enumerate(zip(suggestions, answers)):
                with cols[i]:
                    # Button to select the suggested question
                    if st.button(q, key=f"q_btn_{i}", use_container_width=True):
                        st.session_state.current_answer = a
                        st.session_state.suggested_q = q
                        # Rerun to clear suggestions and show answer
                        st.rerun() 
        else:
            st.info("No close matches found in the FAQ dataset.")
            st.session_state.current_answer = None
            st.session_state.suggested_q = None
            
    # --- Display Selected Answer ---
    if st.session_state.current_answer:
        st.divider()
        st.markdown(f"**Selected Question:** *{st.session_state.suggested_q}*")
        st.success("### Answer")
        st.info(st.session_state.current_answer)
        
# Run the application
if __name__ == "__main__":
    faq_app()




