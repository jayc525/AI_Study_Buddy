from study_buddy.ollama_client import OllamaClient
from study_buddy.rag_pipeline import RAGPipeline


class StudyChatbot:
    """RAG chatbot that can use a local Ollama LLM with no API key.

    Supports conversation memory so follow-up questions work naturally.
    """

    MAX_HISTORY_TURNS = 3

    def __init__(self, document_text: str, model_name: str = "phi3"):
        self.rag = RAGPipeline(document_text)
        self.ollama = OllamaClient(model_name=model_name)
        self.model_name = model_name
        self.conversation_history = []

    def answer(self, question: str) -> str:
        retrieved_chunks = self.rag.retrieve(question, top_k=3)
        if not retrieved_chunks:
            return "I could not find enough notes text to answer from."

        prompt = self._build_prompt_with_history(question, retrieved_chunks)

        if self.ollama.is_available():
            try:
                answer = self.ollama.generate(prompt)
                self._add_to_history(question, answer)
                sources = self._format_sources(retrieved_chunks)
                return f"{answer}\n\n**Retrieved note sources:**\n{sources}"
            except Exception as error:
                return self._error_answer(retrieved_chunks, error)

        return self._fallback_answer(retrieved_chunks)

    def _build_prompt_with_history(self, question, retrieved_chunks):
        """Build prompt with conversation history for follow-up support."""
        context = "\n\n".join(
            f"Source {index}: {chunk.text[:900]}"
            for index, chunk in enumerate(retrieved_chunks, start=1)
        )

        history_text = ""
        if self.conversation_history:
            recent = self.conversation_history[-self.MAX_HISTORY_TURNS:]
            history_lines = []
            for turn in recent:
                history_lines.append(f"Student: {turn['question']}")
                history_lines.append(f"You: {turn['answer'][:300]}")
            history_text = (
                "\nPrevious conversation:\n"
                + "\n".join(history_lines)
                + "\n"
            )

        return f"""
Answer using only the uploaded PDF context.
Keep it short and student-friendly.
If the context does not contain the answer, say the notes do not contain enough information.
Do not show internal reasoning.
{history_text}
Context:
{context}

Question:
{question}

Answer:
""".strip()

    def _add_to_history(self, question, answer):
        """Store conversation turn for follow-up context."""
        self.conversation_history.append({
            "question": question,
            "answer": answer,
        })
        # Keep only recent turns
        if len(self.conversation_history) > self.MAX_HISTORY_TURNS * 2:
            self.conversation_history = self.conversation_history[-self.MAX_HISTORY_TURNS:]

    def status(self) -> str:
        if self.ollama.is_available():
            return f"Local LLM connected through Ollama. Model: {self.model_name}. {self.rag.stats()}"
        return (
            f"Ollama is not running on localhost:11434, so the app is using retrieval fallback. "
            f"Install/open Ollama, then run `ollama pull {self.model_name}` for full local LLM answers. "
            f"No API key is required. {self.rag.stats()}"
        )

    def _format_sources(self, retrieved_chunks) -> str:
        return "\n".join(
            f"- Source {index} | similarity {chunk.score:.2f}: {chunk.text[:220]}..."
            for index, chunk in enumerate(retrieved_chunks, start=1)
        )

    def _fallback_answer(self, retrieved_chunks, error=None) -> str:
        sources = self._format_sources(retrieved_chunks)
        message = (
            "**Ollama is not available right now.**\n\n"
            "This means the local LLM server is not running on `localhost:11434`, "
            "or the selected model has not been downloaded yet. No API key is needed, "
            "but you must install/open Ollama and pull a model first.\n\n"
            f"Run this in a new terminal:\n\n```bash\nollama pull {self.model_name}\n```\n\n"
            "After that, restart this Streamlit app. For now, I am showing the most relevant notes found by RAG retrieval.\n\n"
            f"**Retrieved note sources:**\n{sources}"
        )
        if error is not None:
            message += f"\n\nOllama error: {error}"
        return message

    def _error_answer(self, retrieved_chunks, error) -> str:
        sources = self._format_sources(retrieved_chunks)
        context_answer = self._context_answer(retrieved_chunks)
        return (
            f"{context_answer}\n\n"
            "**Note:** Ollama is running, but the selected model could not finish a clean final answer in time.\n\n"
            f"Model selected: `{self.model_name}`\n\n"
            f"Error: `{error}`\n\n"
            "For faster local LLM answers, run `ollama pull phi3` and select `phi3` in the sidebar.\n\n"
            f"**Retrieved note sources:**\n{sources}"
        )

    def _context_answer(self, retrieved_chunks) -> str:
        best_text = retrieved_chunks[0].text.strip()
        sentences = best_text.split(". ")
        short_answer = ". ".join(sentences[:3]).strip()
        if short_answer and not short_answer.endswith("."):
            short_answer += "."
        return f"Based on the uploaded notes:\n\n{short_answer}"
