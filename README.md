# 🚀 AI Hackathon - Unified Chat, PDF RAG, and Image Analysis App

This project demonstrates a unified Streamlit application for conversational AI, integrating general chat, PDF retrieval-augmented generation (RAG), and image analysis capabilities. It leverages Azure OpenAI for text processing, Ollama for image analysis and embeddings, and Chroma vector database for persistent storage.

## App in Action

![GIF](assets/ollama_streamlit.gif)

## Features

- **Unified Interface**: Single app with three modes - Chat, PDF & Chat, and Image Analysis
- **Conversational AI**: Persistent chat history and intelligent responses across all modes
- **PDF RAG**: Upload and index PDFs, perform exact part-number search and semantic retrieval
- **Intelligent Fallback**: Seamlessly blends PDF knowledge with general chat for comprehensive answers
- **Image Analysis**: Analyze uploaded images using Ollama vision models
- **Vector Storage**: Chroma database for persistent PDF indexing and fast retrieval
- **Azure OpenAI Integration**: Reliable text processing for chat and RAG
- **Ollama Integration**: Local embeddings and vision capabilities

## Installation

Before running the app, ensure you have Python installed on your machine. Then, clone this repository and install the required packages:

```bash
git clone https://github.com/nikhilsamatkar1903/AI-Hackathon---Unified-Chat-PDF-RAG-and-Image-Analysis-App
cd ollama_streamlit_demos
pip install -r requirements.txt
```

## Configuration

1. Copy the `.env.example` file to `.env` and update it with your Azure OpenAI credentials:
   - `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
   - `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
   - `AZURE_OPENAI_DEPLOYMENT`: Deployment name (e.g., gpt-4o-mini)

2. Ensure Ollama is installed and running for embeddings and image analysis.

## Usage

To start the app, run:

```bash
streamlit run run.py
```

Or directly:

```bash
streamlit run src/app/main.py
```

Navigate to the provided URL in your browser to interact with the app.

## Required Models

- **Ollama Models** (for embeddings and vision):
  - Embeddings: `nomic-embed-text`
  - Vision: Any vision-capable model like `llava` or `bakllava`

- **Azure OpenAI**: Deployment for text processing (e.g., gpt-4o-mini)

Ensure Ollama daemon is running and models are pulled:

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llava  # or your preferred vision model
```

## Modes

1. **💬 Chat**: General conversational AI using Azure OpenAI
2. **📄 PDF & Chat**: Ask questions about indexed PDFs or general topics with intelligent fallback
3. **🌋 Image Analysis**: Upload images and ask questions using Ollama vision models

## Testing

Run unit tests:

```bash
pytest -q
```

## Contributing

Interested in contributing?

- Great! I welcome contributions from everyone.

Got questions or suggestions?

- Feel free to open an issue or submit a pull request.

## Acknowledgments

👏 Kudos to the [Ollama](https://ollama.com/) team and [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service/) for their efforts in making AI accessible!
