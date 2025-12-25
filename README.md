# 📚 Book Recommender

An intelligent book recommendation system powered by AI that uses semantic search, emotion analysis, and natural language processing to help users discover their next great read.

## 🌟 Features

- **Semantic Search**:  Uses vector embeddings to understand the meaning behind user queries and find relevant books
- **Emotion Analysis**:  Analyzes book descriptions to classify emotional tones (joy, sadness, fear, anger, surprise, disgust, neutral)
- **Category Classification**: Automatically categorizes books into fiction, non-fiction, children's fiction, and children's non-fiction
- **Interactive Dashboard**: Built with Gradio for an intuitive user experience
- **Advanced Filtering**: Filter recommendations by category and emotional tone
- **Visual Gallery**: Browse books with cover images and detailed descriptions

## 🛠️ Technologies Used

- **Python**:  Core programming language
- **LangChain**: For document processing and vector search
- **OpenAI Embeddings**: For semantic understanding
- **Sentence-transformers**: For semantic understanding
- **HuggingFace Transformers**: For emotion classification and zero-shot categorization
- **Chroma**:  Vector database for similarity search
- **Gradio**:  Interactive web interface
- **Pandas & NumPy**: Data manipulation and analysis

## 📋 Prerequisites

- Python 3.10+
- OpenAI API key (stored in `.env` file)
- Hugging Face API key (stored in `.env` file)
- GPU recommended for optimal performance (transformers models)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/SBM004/Book_recomender.git
cd Book_recomender
```

2. Install required dependencies:
```bash
pip install pandas numpy python-dotenv langchain-community langchain-openai langchain-chroma transformers gradio tqdm
```

3. Create a `.env` file in the root directory:
```
HUGGING_FACE_TOKEN=your_hugging_face_token_key
OPENAI_API_KEY=your_api_key_here (if OpenAIEmbedding is used)

```

## 📊 Data Processing Pipeline

### 1. Data Exploration (`data-exploration.ipynb`)
- Initial analysis of the books dataset
- Data cleaning and preprocessing

### 2. Vector Search Setup (`vector_search.ipynb`)
- Creates tagged descriptions for each book
- Generates embeddings using HuggingFace's sentence-transformers model
- Builds a Chroma vector database for semantic search
- Implements zero-shot classification for missing categories
- Validates category predictions with ~600 samples

### 3. Sentiment Analysis (`sentiment_analysis.ipynb`)
- Analyzes emotional content of book descriptions
- Uses `j-hartmann/emotion-english-distilroberta-base` model
- Extracts 7 emotion scores:  anger, disgust, fear, joy, neutral, sadness, surprise
- Processes descriptions sentence-by-sentence and aggregates maximum scores

## 🎮 Usage

### Running the Dashboard

```bash
python dashboard.py
```

The dashboard will launch in your browser with:
- A text input for describing the type of book you want
- Category dropdown (All, Fiction, Non-fiction, Children's Fiction, Children's Non-fiction)
- Emotional tone filter (All, Happy, Surprising, Angry, Suspenseful, Sad)
- Visual gallery of recommendations

### Example Queries

- "A story about forgiveness"
- "Adventure in space with aliens"
- "Books about teaching children about nature"
- "Mystery thriller with unexpected twists"

## 📁 Project Structure

```
Book_recomender/
├── dashboard.py                 # Main Gradio application
├── data-exploration.ipynb       # Initial data analysis
├── vector_search.ipynb          # Vector database creation & category classification
├── sentiment_analysis.ipynb     # Emotion analysis pipeline
├── books_cleaned.csv            # Cleaned book dataset
├── books_with_emotion.csv       # Dataset with emotion scores
├── tagged_description.txt       # Processed descriptions for vector search
├── cover-not-found.jpg          # Placeholder image
├── . gitignore                   # Git ignore file
└── . env                         # Environment variables (not tracked)
```

## 🔍 How It Works

1. **Query Processing**: User enters a natural language description of desired book
2. **Semantic Search**: Query is embedded and compared against book descriptions in vector database
3. **Initial Filtering**: Top 50 most similar books are retrieved
4. **Category Filtering**: If specified, results are filtered by category
5. **Emotion Sorting**: If an emotional tone is selected, books are sorted by that emotion score
6. **Display**: Top 16 recommendations are displayed with covers and descriptions

## 🎨 Features in Detail

### Semantic Search
- Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model
- Understands context and meaning, not just keywords
- Returns semantically similar books even with different wording

### Emotion Classification
- Processes book descriptions at sentence level
- Calculates maximum emotion scores across all sentences
- Enables filtering by emotional tone preferences

### Category Classification
- Uses Facebook's BART model for zero-shot classification
- Automatically categorizes books with missing metadata
- Achieves high accuracy through validation testing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 

## 📝 License

This project is open source and available under the MIT License. 

## 👤 Author

**SBM004**
- GitHub: [@SBM004](https://github.com/SBM004)

## 🙏 Acknowledgments

- Book data sourced from Kaggle dylanjcastillo/7k-books-with-metadata
- HuggingFace for transformer models
- LangChain for document processing framework
- Gradio for the interactive interface

## 📧 Contact

For questions or feedback, please open an issue on GitHub. 

---

**Happy Reading! 📖✨**
