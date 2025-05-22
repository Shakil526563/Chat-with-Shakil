
````markdown
# Shakil Rana Chatbot

A conversational AI chatbot that answers questions based on the bio and documents of Shakil Rana.  
Built using LangChain, HuggingFace embeddings, FAISS vector store, and Streamlit for a web interface.

## Features

- Load and process PDF documents with LangChain
- Split documents into chunks for better context handling
- Generate embeddings using HuggingFace sentence-transformers
- Store embeddings in FAISS vector database for fast similarity search
- Use ChatGroq LLM for generating context-aware responses
- Interactive Streamlit UI for user queries

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/shakil-rana-chatbot.git
cd shakil-rana-chatbot
````

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv env
source env/bin/activate   # On Windows use `env\Scripts\activate`
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add your `.env` file if needed for API keys or environment variables.

## Usage

Run the chatbot locally:

```bash
streamlit run main.py
```

Open the URL shown in your terminal (usually [http://localhost:8501](http://localhost:8501)) and start chatting!

## Project Structure

* `main.py` — main app logic and Streamlit UI
* `Shakil Rana.pdf` — source bio document
* `requirements.txt` — Python dependencies

## Technologies Used

* Python 3.8+
* LangChain
* HuggingFace Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
* FAISS
* ChatGroq LLM
* Streamlit

## Contribution

Feel free to fork the repo and submit pull requests. For major changes, please open an issue first to discuss.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

Shakil Rana

```
