# AI Study Buddy - Local GenAI RAG Chatbot

AI Study Buddy is a Python GenAI project where a student can upload a PDF, generate Markdown notes, ask doubts through a local LLM chatbot, and create quiz questions from the uploaded content.

The project uses a local RAG architecture with Ollama. It does not require OpenAI, Gemini, or any paid API key. The LLM runs locally on the laptop through Ollama, while the app performs PDF ingestion, text chunking, retrieval, prompt construction, answer generation, notes generation, and quiz generation.

## Features

- Upload a PDF file
- Extract readable text from the PDF
- Build a RAG knowledge base from uploaded notes
- Use a local Ollama LLM such as Llama 3.2, Mistral, Gemma, or Phi
- Generate Markdown notes in bullet points
- Ask questions with grounded answers from retrieved PDF chunks
- Generate multiple-choice quiz questions
- Show retrieved source chunks for answer transparency
- Simple Streamlit interface

## Project Structure

```text
ai-study-buddy/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- CITATIONS.md
|-- study_buddy/
|   |-- __init__.py
|   |-- chatbot.py
|   |-- ollama_client.py
|   |-- pdf_reader.py
|   |-- quiz_generator.py
|   |-- rag_pipeline.py
|   |-- summarizer.py
|   |-- text_utils.py
```

## Model Used

The project uses a local open-source LLM through Ollama. The default model is:

```text
phi3
```

You can also choose:

```text
qwen3:4b
llama3.2
mistral
gemma2:2b
```

The model is not called through an API key. It runs locally through Ollama on your laptop.

## How To Run

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate it.

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install Ollama from:

```text
https://ollama.com/
```

5. Pull a local model:

```bash
ollama pull phi3
```

You can also use:

```bash
ollama pull qwen3:4b
ollama pull llama3.2
ollama pull mistral
ollama pull gemma2:2b
```

6. Start the app:

```bash
streamlit run app.py
```

7. Open the local URL shown in your terminal, usually:

```text
http://localhost:8501
```

## Important Note

This is a local GenAI RAG study assistant. The application code, RAG pipeline, PDF processing, notes generation, and quiz generation are built in Python. The LLM itself is an open-source model served locally by Ollama.

If Ollama is not running, the app still works in retrieval fallback mode and shows the most relevant note chunks.

## Resume Description

Built an AI Study Buddy using Python, Streamlit, Ollama, and RAG. The system ingests PDF notes, extracts and cleans text, chunks the document, retrieves relevant context using TF-IDF and cosine similarity, and generates grounded answers using a local open-source LLM without any paid API key. It also creates Markdown notes and quizzes from the uploaded material.

## Dataset And Citations

No external training dataset is included in this project. The app uses only the PDF uploaded by the student during runtime. See `data/CITATIONS.md` for library, model, and data-source notes.
