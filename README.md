# ScholarPath AI 🎓
**Scholarship Eligibility Assistant**

ScholarPath AI is an intelligent, retrieval-based chatbot that helps university students find scholarships they are eligible for. It uses Natural Language Processing (NLP) to index university scholarship policies and provides grounded, personalized answers based on a student's profile (Major, GPA, Income, etc.).

## 🌟 Unique Features
- **Contextual Student Profile Injection:** Instead of just asking generic questions, students input their GPA, Major, Income, and Year into the sidebar. The AI automatically cross-references this with the scholarship rules!
- **Source Citations:** Every answer comes with an expandable "View Sources" section, showing the exact policy document and excerpt used.
- **Admin Knowledge Base Updater:** New scholarships? Just drop a text file into the `data/` folder and hit "Re-index Knowledge Base" in the sidebar!
- **Local Embeddings:** Uses HuggingFace `sentence-transformers` for fast, free, and local vector embeddings. 

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **LLM Engine:** Google Gemini (via `langchain-google-genai`)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Orchestration:** LangChain

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Setup Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add API Key
Copy the `.env.example` file to `.env` and add your Google Gemini API key:
```bash
cp .env.example .env
```
Open `.env` and set:
`GOOGLE_API_KEY=your_actual_key_here`

*(You can get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey))*

### 5. Build Knowledge Base
I've already included 3 dummy scholarship documents in the `data/` folder.
Run the vector indexing script to build the FAISS database:
```bash
python vector_store.py
```
*(This will download the HuggingFace embedding model and create a `faiss_index` folder.)*

### 6. Run the App
```bash
streamlit run app.py
```
Enjoy exploring ScholarPath AI!
