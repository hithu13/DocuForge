import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="DocuForge",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# CUSTOM STYLE
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">📚 DocuForge</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Turn your documents into an interactive learning experience.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pages" not in st.session_state:
    st.session_state.pages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "collection" not in st.session_state:
    st.session_state.collection = None

if "model" not in st.session_state:
    st.session_state.model = None

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader(
        "📂 Upload PDF",
        type=["pdf"]
    )

    answer_mode = st.selectbox(
        "🎯 Answer Mode",
        [
            "😊 Simple Answer",
            "🎓 Exam Answer",
            "📚 Detailed Answer"
        ]
    )

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file:

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:

        pages.append(
            page.extract_text() or ""
        )

    text = "\n".join(pages)

    st.session_state.pages = pages


    # =====================================================
    # CREATE CHUNKS
    # =====================================================

    chunk_size = 500

    chunks = []
    chunk_pages = []

    for page_number, page_text in enumerate(
        pages,
        start=1
    ):

        for i in range(
            0,
            len(page_text),
            chunk_size
        ):

            chunks.append(
                page_text[i:i + chunk_size]
            )

            chunk_pages.append(
                page_number
            )

    st.session_state.chunks = chunks


    # =====================================================
    # LOAD MODEL
    # =====================================================

    model = load_model()

    st.session_state.model = model


    # =====================================================
    # CREATE EMBEDDINGS
    # =====================================================

    with st.spinner(
        "🧠 Processing document..."
    ):

        embeddings = model.encode(
            chunks
        ).tolist()


    # =====================================================
    # CHROMADB
    # =====================================================

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="documents"
    )


    collection.upsert(

        documents=chunks,

        embeddings=embeddings,

        ids=[
            str(i)
            for i in range(len(chunks))
        ],

        metadatas=[
            {"page": page}
            for page in chunk_pages
        ]
    )


    st.session_state.collection = collection

    st.session_state.document_ready = True


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📄 Document",
        "💬 Ask",
        "📊 Visualize",
        "ℹ️ About"
    ]
)


# =========================================================
# TAB 1 — DOCUMENT
# =========================================================

with tab1:

    st.header("📄 Document")

    if uploaded_file and st.session_state.document_ready:

        st.success(
            "✅ Document uploaded successfully!"
        )


        # Document information
        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "📄 Pages",
                len(pages)
            )

        with col2:

            st.metric(
                "📝 Characters",
                len(text)
            )

        with col3:

            st.metric(
                "✂️ Chunks",
                len(chunks)
            )


        # Page viewer
        st.subheader("👁️ View Document")

        page_number = st.selectbox(
            "Select a page:",
            range(1, len(pages) + 1)
        )

        st.text_area(
            "Page content:",
            pages[page_number - 1],
            height=300
        )


        # Search document
        st.subheader("🔎 Search in Document")

        search_word = st.text_input(
            "Enter a word or topic:"
        )


        if search_word:

            found_pages = []

            for i, page_text in enumerate(pages):

                if search_word.lower() in page_text.lower():

                    found_pages.append(i + 1)


            if found_pages:

                st.success(
                    "Found on page(s): "
                    + ", ".join(
                        map(str, found_pages)
                    )
                )


                for page in found_pages:

                    with st.expander(
                        f"📄 Page {page}"
                    ):

                        st.write(
                            pages[page - 1]
                        )

            else:

                st.warning(
                    "❌ Word/topic not found."
                )

    else:

        st.info(
            "👈 Upload a PDF from the sidebar "
            "to view the document."
        )


# =========================================================
# TAB 2 — ASK
# =========================================================

with tab2:

    st.header("💬 Ask Your Document")

    if (
        uploaded_file
        and st.session_state.document_ready
    ):

        st.write(
            "Ask questions based on the uploaded document."
        )


        # Display previous messages
        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.write(
                    message["content"]
                )


        # Question input
        question = st.chat_input(
            "💬 Ask a question..."
        )


        if question:

            # Show question
            with st.chat_message("user"):

                st.write(question)


            st.session_state.messages.append({

                "role": "user",

                "content": question

            })


            # =================================================
            # SEARCH RELEVANT INFORMATION
            # =================================================

            with st.spinner(
                "🔍 Searching document..."
            ):

                model = st.session_state.model

                collection = st.session_state.collection


                question_embedding = model.encode(
                    [question]
                ).tolist()


                results = collection.query(

                    query_embeddings=question_embedding,

                    n_results=3
                )


                relevant_text = "\n\n".join(
                    results["documents"][0]
                )


                relevant_pages = []


                for metadata in results["metadatas"][0]:

                    page = metadata["page"]


                    if page not in relevant_pages:

                        relevant_pages.append(
                            page
                        )


            # =================================================
            # ANSWER MODE
            # =================================================

            if answer_mode == "😊 Simple Answer":

                mode_instruction = """
Give an easy-to-understand 6-mark answer.

Structure the answer like this:

1. Start with a simple definition or introduction.
2. Explain the main concept clearly.
3. Give the important points in bullet points or
   small paragraphs.
4. Include a simple example if it is useful.
5. End with a short conclusion if needed.

Use simple words and short sentences.

The answer should have enough content for a
6-mark question, but do not add unnecessary
information.

Make it easy for a student to understand and remember.
"""


            elif answer_mode == "🎓 Exam Answer":

                mode_instruction = """
Give an exam-ready answer.

Start with a definition.
Then give the important points in an organized format.
Use bullet points where useful.
Include an example if needed.

Keep the answer focused and suitable for writing
in an examination.
"""


            else:

                mode_instruction = """
Give a detailed explanation.

Explain the concept step by step.
Include important points, examples, and details
from the document.

Make the explanation complete and easy to follow.
"""


            # =================================================
            # OLLAMA
            # =================================================

            with st.spinner(
                "🤖 Generating answer..."
            ):

                prompt = f"""
You are DocuForge, a document learning assistant.

Answer the question using the uploaded document.

{mode_instruction}

Use related information from the document
even if the exact words of the question
are different.

Do not make up information that is not supported
by the document.

If there is not enough information, say:

"The document does not contain enough information
to answer this question."

Document information:

{relevant_text}

Question:

{question}
"""


                response = ollama.chat(

                    model="llama3.2:3b",

                    messages=[

                        {
                            "role": "user",
                            "content": prompt
                        }

                    ]
                )


                answer = response[
                    "message"
                ][
                    "content"
                ]


            # =================================================
            # DISPLAY ANSWER
            # =================================================

            with st.chat_message("assistant"):

                st.write(answer)

                st.caption(
                    "📄 Relevant page(s): "
                    + ", ".join(
                        map(
                            str,
                            relevant_pages
                        )
                    )
                )


            # Save answer
            st.session_state.messages.append({

                "role": "assistant",

                "content": answer

            })


    else:

        st.info(
            "👈 Upload a PDF first."
        )


# =========================================================
# TAB 3 — VISUALIZE
# =========================================================

with tab3:

    st.header("📊 Visualize")

    if (
        uploaded_file
        and st.session_state.document_ready
    ):

        st.write(
            "Understand how DocuForge processes your document."
        )


        st.subheader(
            "🔄 DocuForge Workflow"
        )


        st.markdown(
            """
            ### 📄 Upload PDF

            ⬇️

            ### ✂️ Split Document into Chunks

            ⬇️

            ### 🧠 Create Embeddings

            ⬇️

            ### 🔎 Retrieve Relevant Information

            ⬇️

            ### 🤖 Generate Answer

            ⬇️

            ### 💬 Display Result
            """
        )


        st.info(
            "This flow shows how the document is processed "
            "before generating an answer."
        )


    else:

        st.info(
            "👈 Upload a PDF first."
        )


# =========================================================
# TAB 4 — ABOUT
# =========================================================

with tab4:

    st.header("ℹ️ About DocuForge")

    st.write(
        """
        **DocuForge** is an interactive document
        learning platform.

        It allows users to upload a PDF, view and
        search the document, ask questions, and
        receive answers based on the document.
        """
    )


    st.subheader("🛠️ Technologies Used")

    st.write(
        """
        • Python
        • Streamlit
        • Ollama
        • Llama 3.2
        • Sentence Transformers
        • ChromaDB
        • PyPDF
        """
    )


    st.subheader("✨ Main Features")

    st.write(
        """
        📄 PDF Upload

        👁️ Page Viewer

        🔎 Document Search

        💬 Question Answering

        😊 Simple 6-Mark Answer

        🎓 Exam Answer

        📚 Detailed Answer

        🧠 Retrieval-Augmented Generation
        """
    )