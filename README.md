# Document RAG ChatBot

## Overview
The Document RAG ChatBot is an AI-powered chatbot designed to process and retrieve information from documents using Retrieval-Augmented Generation (RAG). It enables users to ask questions and receive precise answers based on document content.

## Features
- Upload and process documents (PDF, TXT, etc.).
- Uses Retrieval-Augmented Generation (RAG) for better responses.
- AI-powered chatbot interface for interactive queries.
- Provides contextual and relevant answers from the document.
- Built-in document search and summarization features.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Virtual environment (optional but recommended)

### Clone the Repository
```sh
git clone https://github.com/krishSharma1810/DocumentRAGChatBot.git
cd DocumentRAGChatBot
```

### Backend Setup
1. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the backend:
   ```sh
   streamlit run app.py
   ```
   
## Usage
1. Upload a document through the Streamlit interface.
2. Ask questions related to the uploaded document.
3. The chatbot retrieves relevant sections and generates responses.

## Technologies Used
- Python (FastAPI/Flask)
- Streamlit (Frontend UI)
- NLP Libraries (spaCy, Transformers, OpenAI API)
- Vector Databases for efficient document search

## Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Commit changes and push to your fork.
4. Submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For queries, contact: [sharmakrish1810work@gmail.com](mailto:sharmakrish1810work@gmail.com)
