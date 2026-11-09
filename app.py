"""
Campus FAQ Chatbot
Modern, interactive Streamlit interface powered by TF-IDF and Cosine Similarity.
"""

from datetime import datetime
import json
import os
import streamlit as st

from chatbot import FAQChatbot

# Page configuration
st.set_page_config(
    page_title="Campus FAQ Assistant | AI Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1100px;
    }

    /* Header banner styling */
    .header-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }
    .header-banner h1 {
        color: #ffffff;
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .header-banner p {
        color: #e0e7ff;
        font-size: 0.95rem;
        margin: 0;
    }
    .demo-badge {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.2);
        color: #ffffff;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* Metadata pill styling */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .pill-category {
        background-color: #e0e7ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
    }
    .pill-score {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .pill-fallback {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .timestamp-text {
        color: #94a3b8;
        font-size: 0.74rem;
        margin-top: 4px;
        display: block;
    }

    /* Suggestion chips container */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0 16px 0;
    }

    /* Sidebar info card */
    .sidebar-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1.2rem;
    }

    /* Streamlit chat message padding */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.6rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_chatbot_instance() -> FAQChatbot:
    """Initialize and cache the FAQ chatbot engine."""
    return FAQChatbot()


# Initialize chatbot
try:
    bot = get_chatbot_instance()
except Exception as e:
    st.error(f"Failed to load FAQ Chatbot engine: {e}")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! 👋 I am the **Campus FAQ Assistant**. Ask me anything about Admissions, Courses, Fees, Scholarships, Hostel, Transportation, Placements, Examinations, Library, or Campus Facilities!",
            "category": "Welcome",
            "similarity_score": None,
            "confidence_label": "System Greeting",
            "is_fallback": False,
            "timestamp": datetime.now().strftime("%I:%M %p"),
            "top_matches": []
        }
    ]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Sidebar Controls
with st.sidebar:
    st.markdown("### 🎓 Campus FAQ Assistant")
    st.caption("Intelligent Knowledge Base • NLP Chatbot")

    st.markdown("---")

    # Category filter
    all_categories = ["All Categories"] + bot.get_categories()
    selected_category = st.selectbox(
        "📂 Filter by FAQ Category:",
        options=all_categories,
        index=0,
        help="Restrict or focus FAQ matching on a specific campus department."
    )
    category_filter_arg = None if selected_category == "All Categories" else selected_category

    # Threshold slider
    similarity_threshold = st.slider(
        "🎯 Similarity Cutoff Threshold:",
        min_value=0.10,
        max_value=0.60,
        value=0.25,
        step=0.05,
        help="Queries with a cosine similarity below this score trigger a fallback response to avoid false answers."
    )

    st.markdown("---")

    # Actions: Clear Chat & Export History
    st.markdown("#### ⚙️ Conversation Controls")
    
    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat history cleared. How can I assist you with campus FAQs today?",
                    "category": "System",
                    "similarity_score": None,
                    "confidence_label": "Reset",
                    "is_fallback": False,
                    "timestamp": datetime.now().strftime("%I:%M %p"),
                    "top_matches": []
                }
            ]
            st.rerun()

    with col_export:
        # Prepare exportable JSON
        chat_export_data = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="💾 Export JSON",
            data=chat_export_data,
            file_name=f"faq_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    # Technical Details Card
    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-card">
            <h5 style="margin-top:0; font-size:0.92rem; color:#1e293b;">🔬 Machine Learning Pipeline</h5>
            <ul style="padding-left:1.2rem; margin-bottom:0; font-size:0.82rem; color:#475569;">
                <li><b>Algorithm:</b> TF-IDF & Cosine Similarity</li>
                <li><b>NLP:</b> Tokenization, Lemmatization, Stopwords</li>
                <li><b>Dataset:</b> 55 Curated Campus FAQs</li>
                <li><b>Reproducibility:</b> Deterministic Vector Space</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


# Main Content Area
tabs = st.tabs(["💬 Campus Chatbot", "📚 Browse All FAQs", "ℹ️ About the Project"])

with tabs[0]:
    # Header Banner
    st.markdown(
        """
        <div class="header-banner">
            <span class="demo-badge">AI-Powered NLP Assistant</span>
            <h1>🎓 Campus FAQ Assistant</h1>
            <p>Instant answers to queries on Admissions, Fees, Courses, Hostels, Placements, and Campus Life using TF-IDF & Cosine Similarity.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Quick Suggestion Chips
    st.markdown("**💡 Quick Suggestions (Click to Ask):**")
    suggested_questions = bot.get_suggested_questions(limit=4, category=category_filter_arg)
    
    cols = st.columns(len(suggested_questions))
    for i, q in enumerate(suggested_questions):
        with cols[i]:
            # Shorten label for button if needed
            btn_label = q if len(q) <= 45 else q[:42] + "..."
            if st.button(f"📌 {btn_label}", key=f"sug_{i}", use_container_width=True, help=q):
                st.session_state.pending_query = q
                st.rerun()

    st.markdown("---")

    # Display Chat History
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar="🧑‍🎓" if role == "user" else "🤖"):
            st.markdown(msg["content"])

            # Show metadata pills for assistant messages with match metrics
            if role == "assistant" and msg.get("similarity_score") is not None:
                cat = msg.get("category")
                score = msg.get("similarity_score")
                conf_label = msg.get("confidence_label", "")
                is_fb = msg.get("is_fallback", False)

                pill_class = "pill-fallback" if is_fb else "pill-score"
                pills_html = f"""
                <div>
                    {f'<span class="metric-pill pill-category">📁 {cat}</span>' if cat else ''}
                    <span class="metric-pill {pill_class}">🎯 Cosine Similarity: {score:.3f} ({conf_label})</span>
                </div>
                """
                st.markdown(pills_html, unsafe_allow_html=True)

                # Show alternative suggestions expander if available
                top_matches = msg.get("top_matches", [])
                if top_matches and len(top_matches) > 1 and not is_fb:
                    with st.expander("🔍 See related questions considered"):
                        for m in top_matches:
                            st.write(f"- **{m['question']}** *({m['category']})* — Score: `{m['similarity_score']:.3f}`")

            # Timestamp
            ts = msg.get("timestamp")
            if ts:
                st.markdown(f'<span class="timestamp-text">⏱️ {ts}</span>', unsafe_allow_html=True)

    # Handle Pending Query or New Chat Input
    user_input = st.chat_input("Type your question here (e.g., 'What is the B.Tech tuition fee?' or 'Hostel rules')...")

    # If clicked suggestion
    if st.session_state.pending_query:
        active_query = st.session_state.pending_query
        st.session_state.pending_query = None
    elif user_input:
        active_query = user_input
    else:
        active_query = None

    if active_query:
        current_time = datetime.now().strftime("%I:%M %p")
        
        # 1. Append User Message
        st.session_state.messages.append({
            "role": "user",
            "content": active_query,
            "timestamp": current_time
        })

        # Display user message immediately
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(active_query)
            st.markdown(f'<span class="timestamp-text">⏱️ {current_time}</span>', unsafe_allow_html=True)

        # 2. Generate Chatbot Response with Loading Spinner
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing query with TF-IDF vectorizer and Cosine Similarity..."):
                response_data = bot.get_response(
                    user_query=active_query,
                    category_filter=category_filter_arg,
                    threshold=similarity_threshold
                )

            st.markdown(response_data["answer"])

            # Metadata display
            if response_data.get("similarity_score") is not None:
                cat = response_data.get("category")
                score = response_data.get("similarity_score", 0.0)
                conf_label = response_data.get("confidence_label", "")
                is_fb = response_data.get("is_fallback", False)

                pill_class = "pill-fallback" if is_fb else "pill-score"
                pills_html = f"""
                <div>
                    {f'<span class="metric-pill pill-category">📁 {cat}</span>' if cat else ''}
                    <span class="metric-pill {pill_class}">🎯 Cosine Similarity: {score:.3f} ({conf_label})</span>
                </div>
                """
                st.markdown(pills_html, unsafe_allow_html=True)

                top_matches = response_data.get("top_matches", [])
                if top_matches and len(top_matches) > 1 and not is_fb:
                    with st.expander("🔍 See related questions considered"):
                        for m in top_matches:
                            st.write(f"- **{m['question']}** *({m['category']})* — Score: `{m['similarity_score']:.3f}`")

            st.markdown(f'<span class="timestamp-text">⏱️ {current_time}</span>', unsafe_allow_html=True)

        # Append assistant response to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data["answer"],
            "category": response_data.get("category"),
            "similarity_score": response_data.get("similarity_score"),
            "confidence_label": response_data.get("confidence_label"),
            "is_fallback": response_data.get("is_fallback", False),
            "timestamp": current_time,
            "top_matches": response_data.get("top_matches", [])
        })

# Tab 2: Browse and Search FAQ Dataset
with tabs[1]:
    st.markdown("### 📚 Explore Campus Knowledge Base")
    st.caption("Browse all 55 verified FAQs across 10 academic and campus categories.")

    col_search, col_cat = st.columns([3, 2])
    with col_search:
        search_query = st.text_input("🔍 Search FAQs by keyword:", placeholder="e.g. scholarship, wifi, hostel...")
    with col_cat:
        browse_cat = st.selectbox("Filter category:", ["All"] + bot.get_categories())

    filtered_faqs = bot.search_faqs(search_query, category=browse_cat) if search_query else (
        bot.faqs if browse_cat == "All" else bot.get_faqs_by_category(browse_cat)
    )

    st.info(f"Showing **{len(filtered_faqs)}** FAQ entries matching criteria.")

    for item in filtered_faqs:
        with st.expander(f"**[{item['category']}]** {item['question']}"):
            st.write(item["answer"])
            if item.get("keywords"):
                st.caption(f"**Keywords:** {', '.join(item['keywords'])}")

# Tab 3: About & Architecture Explanation
with tabs[2]:
    st.markdown("### 🏛️ About Campus FAQ Chatbot")
    st.markdown(
        """
        An intelligent campus FAQ assistant designed to provide fast, reliable answers to common university questions across academic and campus departments.

        #### 🚀 Key Features
        - **NLP Preprocessing Pipeline:** Tokenization, lowercasing, punctuation stripping, stopword removal, and WordNet lemmatization with automatic fallback mechanisms.
        - **TF-IDF Vectorization:** Transforms FAQ questions and queries into normalized numerical feature vectors with unigrams and bigrams.
        - **Cosine Similarity Matching:** Measures the cosine angle between query and FAQ vectors in high-dimensional vector space.
        - **Safety Thresholding:** Protects users by returning a polite fallback message when similarity drops below the threshold, preventing hallucinated answers.
        - **Interactive Streamlit UI:** Complete with user and assistant message bubbles, suggestion chips, category filters, session export, and timestamps.
        """
    )
