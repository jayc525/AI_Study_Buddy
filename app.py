import streamlit as st

from study_buddy.chatbot import StudyChatbot
from study_buddy.pdf_reader import extract_text_from_pdf
from study_buddy.quiz_generator import generate_quiz, generate_llm_quiz
from study_buddy.summarizer import create_notes_markdown, create_llm_notes
from study_buddy.text_utils import clean_text


st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="wide",
)


def reset_study_state() -> None:
    st.session_state.document_text = ""
    st.session_state.notes_markdown = ""
    st.session_state.quiz_count = 0
    st.session_state.model_name = ""
    st.session_state.chatbot = None
    st.session_state.quiz = []
    st.session_state.chat_history = []
    st.session_state.score_history = []
    st.session_state.use_llm_notes = False
    st.session_state.use_llm_quiz = False


if "document_text" not in st.session_state:
    reset_study_state()


st.title("📚 AI Study Buddy")
st.caption("Local GenAI + RAG study assistant using Ollama. Upload a PDF, chat with notes, generate notes, and create quizzes.")

with st.sidebar:
    st.header("📄 Upload Notes")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    model_name = st.selectbox("Ollama model", ["phi3", "qwen3:4b", "llama3.2", "mistral", "gemma2:2b"], index=0)
    notes_length = st.slider("Number of note points", min_value=3, max_value=12, value=6)
    quiz_count = st.slider("Quiz questions", min_value=3, max_value=10, value=5)

    st.divider()
    st.subheader("⚡ AI Features")
    use_llm_notes = st.toggle("AI-Generated Notes", value=st.session_state.get("use_llm_notes", False),
                               help="Use Ollama to rephrase notes instead of extracting sentences")
    use_llm_quiz = st.toggle("AI-Generated Quiz", value=st.session_state.get("use_llm_quiz", False),
                              help="Use Ollama to create conceptual questions instead of fill-in-the-blank")
    st.session_state.use_llm_notes = use_llm_notes
    st.session_state.use_llm_quiz = use_llm_quiz

    st.divider()
    if st.button("🗑️ Clear current notes"):
        reset_study_state()
        st.rerun()

if uploaded_file is not None:
    with st.spinner("Reading your PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        cleaned_text = clean_text(raw_text)

    if cleaned_text and (
        cleaned_text != st.session_state.document_text
        or model_name != st.session_state.model_name
    ):
        st.session_state.document_text = cleaned_text
        st.session_state.model_name = model_name
        with st.spinner("Building semantic index... (first time may take a moment)"):
            st.session_state.chatbot = StudyChatbot(cleaned_text, model_name=model_name)
        st.session_state.quiz_count = 0
        st.session_state.chat_history = []

    if cleaned_text:
        # Generate notes
        if use_llm_notes:
            with st.spinner("Generating AI notes with Ollama..."):
                llm_notes = create_llm_notes(cleaned_text, model_name=model_name, point_count=notes_length)
            st.session_state.notes_markdown = llm_notes or create_notes_markdown(cleaned_text, point_count=notes_length)
        else:
            st.session_state.notes_markdown = create_notes_markdown(cleaned_text, point_count=notes_length)

        # Generate quiz
        if quiz_count != st.session_state.quiz_count:
            if use_llm_quiz:
                with st.spinner("Generating AI quiz with Ollama..."):
                    llm_quiz = generate_llm_quiz(cleaned_text, model_name=model_name, question_count=quiz_count)
                st.session_state.quiz = llm_quiz or generate_quiz(cleaned_text, question_count=quiz_count)
            else:
                st.session_state.quiz = generate_quiz(cleaned_text, question_count=quiz_count)
            st.session_state.quiz_count = quiz_count

if not st.session_state.document_text:
    st.info("Upload a PDF from the sidebar to begin.")
    st.stop()

notes_tab, chat_tab, quiz_tab, flashcard_tab, rag_tab, raw_text_tab = st.tabs(
    ["📝 Notes", "💬 Chat", "❓ Quiz", "🃏 Flashcards", "🔧 RAG + LLM", "📄 Raw Text"]
)

with notes_tab:
    st.markdown(st.session_state.notes_markdown)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Notes (.md)",
            data=st.session_state.notes_markdown,
            file_name="study_notes.md",
            mime="text/markdown",
        )
    with col2:
        if st.button("🔄 Regenerate Notes"):
            if use_llm_notes:
                with st.spinner("Regenerating AI notes..."):
                    llm_notes = create_llm_notes(
                        st.session_state.document_text, model_name=model_name, point_count=notes_length
                    )
                st.session_state.notes_markdown = llm_notes or create_notes_markdown(
                    st.session_state.document_text, point_count=notes_length
                )
            else:
                st.session_state.notes_markdown = create_notes_markdown(
                    st.session_state.document_text, point_count=notes_length
                )
            st.rerun()

with chat_tab:
    st.subheader("Ask Doubts")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Ask a question about your uploaded PDF")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.spinner("Generating answer with local Ollama LLM..."):
            answer = st.session_state.chatbot.answer(user_question)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

with quiz_tab:
    st.subheader("Quiz From Notes")

    # Progress tracking
    if st.session_state.score_history:
        correct = sum(st.session_state.score_history)
        total = len(st.session_state.score_history)
        st.progress(correct / total, text=f"Score: {correct}/{total} correct ({100 * correct // total}%)")

    if not st.session_state.quiz:
        st.warning("Not enough clean text was found to create quiz questions.")
    else:
        for index, item in enumerate(st.session_state.quiz, start=1):
            st.markdown(f"**Q{index}. {item['question']}**")
            selected = st.radio(
                "Choose one answer",
                item["options"],
                key=f"quiz_{index}",
                label_visibility="collapsed",
            )

            if st.button(f"Check Q{index}", key=f"check_{index}"):
                if selected == item["answer"]:
                    st.success("✅ Correct!")
                    st.session_state.score_history.append(1)
                else:
                    st.error(f"❌ Not quite. Correct answer: {item['answer']}")
                    st.session_state.score_history.append(0)

            with st.expander("Show source"):
                st.write(item["source"])

        # Export quiz
        quiz_text = "\n\n".join(
            f"Q{i}. {q['question']}\n"
            + "\n".join(f"  {chr(65+j)}) {opt}" for j, opt in enumerate(q["options"]))
            + f"\n  Answer: {q['answer']}"
            for i, q in enumerate(st.session_state.quiz, start=1)
        )
        st.download_button(
            "📥 Download Quiz (.txt)",
            data=quiz_text,
            file_name="study_quiz.txt",
            mime="text/plain",
        )

with flashcard_tab:
    st.subheader("🃏 Quick Flashcards")
    st.caption("Key terms extracted from your notes")

    from study_buddy.text_utils import important_words
    from collections import Counter

    terms = Counter(important_words(st.session_state.document_text)).most_common(15)

    if not terms:
        st.warning("Not enough text to generate flashcards.")
    else:
        # Build simple term → context flashcards
        sentences = st.session_state.document_text.split(". ")
        for idx, (term, freq) in enumerate(terms):
            # Find the sentence containing this term
            context = ""
            for sent in sentences:
                if term in sent.lower():
                    context = sent.strip()
                    if not context.endswith("."):
                        context += "."
                    break

            if context:
                with st.expander(f"**{term.title()}** (appears {freq}x) — click to reveal"):
                    st.write(context)

with rag_tab:
    st.subheader("RAG + Local LLM")
    st.write(st.session_state.chatbot.status())
    st.markdown(
        "- LLM runtime: Ollama running locally\n"
        "- LLM models: Llama 3.2, Mistral, Gemma, or Phi depending on what you install\n"
        "- RAG retrieval: Semantic embeddings (FAISS + MiniLM) with TF-IDF fallback\n"
        "- Context source: only the uploaded PDF\n"
        "- API key: not required\n"
        "- Purpose: grounded answers, notes, and study support"
    )

with raw_text_tab:
    st.subheader("Raw Extracted Text")
    st.warning(
        "If this text has many spelling mistakes, the PDF is probably scanned, handwritten, "
        "or using a font that normal PDF extraction cannot read perfectly."
    )
    st.text_area("Text found in PDF", st.session_state.document_text, height=500)
