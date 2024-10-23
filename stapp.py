import os
import pdfplumber
import streamlit as st
from collections import Counter
from pymongo import MongoClient
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text() or ""  # Handle pages with no text
            text += page_text
    return text, num_pages

# Function to summarize text
def custom_summarization(text, num_sentences=3):
    """Summarize the text by selecting the top N ranked sentences."""
    if not text.strip():
        return ""  # Return empty if there's no text

    sentences = sent_tokenize(text)  # Tokenize text into sentences
    words = word_tokenize(text.lower())  # Tokenize and lower the text
    word_freq = Counter(words)  # Count word frequencies

    # Rank sentences based on the frequency of words
    ranked_sentences = sorted(
        sentences,
        key=lambda sentence: sum(word_freq[word] for word in word_tokenize(sentence.lower())),
        reverse=True
    )
    
    return " ".join(ranked_sentences[:num_sentences])  # Return top N sentences

# Function to remove stop words from text
def remove_stopwords(text):
    """Remove stop words from the input text."""
    stop_words = set(stopwords.words('english'))  # Get English stop words
    filtered_words = [word for word in text.split() if word.lower() not in stop_words]
    return " ".join(filtered_words)

# Function to extract top N keywords based on frequency
def extract_keywords(text, top_n=5):
    if not text.strip():
        return []

    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words]).lower()
    words = word_tokenize(text)
    word_freq = Counter(words)
    
    return [word for word, freq in word_freq.most_common(top_n) if word.isalpha()]

# Function to save data to MongoDB
def save_to_mongo(pdf_path, summary, keywords, db_name, collection_name):
    try:
        client = MongoClient('localhost', 27017)
        db = client[db_name]
        collection = db[collection_name]
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        collection.insert_one({'pdf_name': pdf_name, 'summary': summary, 'keywords': keywords})
    except Exception as e:
        st.error(f"Failed to save to MongoDB: {str(e)}")
# Function to process multiple PDFs in a folder and save summaries and keywords to MongoDB
def run_pipeline(folder_path, db_name, collection_name):
    """Process all PDF files in the specified folder, extract summaries, keywords, and save to MongoDB."""
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        st.warning("No PDF files found in the folder.")
        return

    # Process each PDF file concurrently
    with ThreadPoolExecutor() as executor:
        futures = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            futures.append(executor.submit(process_single_pdf, pdf_path, db_name, collection_name))
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()  # This will raise any exception encountered in a task

# Function to process a single PDF file and save summary and keywords to MongoDB
def process_single_pdf(pdf_path, db_name, collection_name):
    """Process a single PDF file and save summary and keywords to MongoDB."""
    try:
        text, num_pages = extract_text_from_pdf(pdf_path)
        st.write(f"Processing '{os.path.basename(pdf_path)}' ({num_pages} pages)")
        
        # Preprocess the text (optional: can add more preprocessing steps)
        processed_text = remove_stopwords(text)
        
        # Generate summary and keywords
        summary = custom_summarization(processed_text)
        keywords = extract_keywords(processed_text)
        
        # Save to MongoDB
        save_to_mongo(pdf_path, summary, keywords, db_name, collection_name)
        st.write(f"Saved summary and keywords for '{os.path.basename(pdf_path)}'")
    
    except Exception as e:
        st.error(f"Error processing '{os.path.basename(pdf_path)}': {str(e)}")


# Streamlit application
def main():
    st.title("PDF Processing Pipeline")
    folder_path = st.text_input("Folder Path:")
    db_name = st.text_input("Database Name:", "pdf_summarization3").replace(" ", "_")
    collection_name = st.text_input("Collection Name:", "summaries").replace(" ", "_")

    if st.button("Process PDFs"):
        if os.path.exists(folder_path):
            try:
                run_pipeline(folder_path, db_name, collection_name)
                st.success("PDFs processed successfully!")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("The folder path does not exist.")

if __name__ == "__main__":
    main()
