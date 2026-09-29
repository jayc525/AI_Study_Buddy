# Dataset And Citation Notes

## Dataset

This project does not include a fixed dataset.

The app processes only the PDF uploaded by the student at runtime. That means the student should upload notes, textbook pages, or study material they are allowed to use.

## Libraries Used

- Streamlit: https://streamlit.io/
- pypdf: https://pypdf.readthedocs.io/
- scikit-learn: https://scikit-learn.org/
- Ollama: https://ollama.com/

## Local LLM Models

The app can use local Ollama models such as:

- Llama 3.2 through Ollama: https://ollama.com/library/llama3.2
- Qwen 3 through Ollama: https://ollama.com/library/qwen3
- Mistral through Ollama: https://ollama.com/library/mistral
- Gemma 2 through Ollama: https://ollama.com/library/gemma2
- Phi-3 through Ollama: https://ollama.com/library/phi3

## Method Notes

- PDF extraction is done using `pypdf`.
- The RAG retrieval pipeline uses TF-IDF and cosine similarity from `scikit-learn`.
- The answer generator uses a local open-source LLM served through Ollama.
- The notes are generated using a local word-frequency extractive notes method.
- The quiz is generated using local rules that turn important terms from sentences into fill-in-the-blank questions.
