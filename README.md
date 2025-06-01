# AI Agent RAG

A Python-based AI agent with Retrieval-Augmented Generation (RAG) capabilities.

## Features

- **Document Processing**: Support for PDF, DOCX, and text files
- **Vector Storage**: ChromaDB integration for efficient similarity search
- **LLM Integration**: OpenAI GPT models for response generation
- **RAG Pipeline**: Complete retrieval-augmented generation workflow
- **API Interface**: FastAPI-based REST API
- **Web Interface**: Streamlit dashboard for easy interaction

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/naveenvivek/aiagentrag.git
   cd aiagentrag
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Run the application**:
   ```bash
   # API server
   python -m uvicorn src.api.main:app --reload
   
   # Streamlit interface
   streamlit run src/ui/app.py
   ```

## Project Structure

```
aiagentrag/
├── src/
│   ├── agents/          # AI agent implementations
│   ├── rag/            # RAG pipeline components
│   ├── vectorstore/    # Vector database operations
│   ├── documents/      # Document processing
│   ├── api/           # FastAPI endpoints
│   ├── ui/            # Streamlit interface
│   └── utils/         # Utility functions
├── data/              # Data storage
├── logs/              # Application logs
├── tests/             # Test cases
└── docs/              # Documentation
```

## Usage

### Basic RAG Query
```python
from src.rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
response = pipeline.query("What is machine learning?")
print(response)
```

### Adding Documents
```python
from src.documents.processor import DocumentProcessor

processor = DocumentProcessor()
processor.add_document("path/to/document.pdf")
```

## Configuration

Edit the `.env` file to configure:
- OpenAI API key
- Vector store settings
- Chunk sizes and overlap
- Model parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License
