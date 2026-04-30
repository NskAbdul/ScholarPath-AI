import streamlit as st
import os
import shutil
import json
import time
from dotenv import load_dotenv
from vector_store import load_vector_store, build_vector_store
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Find Scholarships",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- SESSION STATE INITIALIZATION ---
if "profile" not in st.session_state:
    st.session_state.profile = {
        "major": "", 
        "gpa": 8.0, 
        "income": 50000, 
        "year": "1st Year",
        "category": "General",
        "language": "English"
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me any questions about the scholarships, deadlines, or required documents."}
    ]

if "eligibility_results" not in st.session_state:
    st.session_state.eligibility_results = None

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

# --- UNIVERSAL NATIVE THEME CSS ---
universal_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div, li, label {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean Cards adapting to native Streamlit background colors */
    .clean-card {
        background: var(--secondary-background-color);
        border-radius: 16px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .result-card {
        background: var(--secondary-background-color);
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(0,0,0,0.1);
    }
    
    /* Hero Banner Styling */
    .hero-container {
        padding: 40px 20px 20px 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 12px;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.8;
        margin-bottom: 8px;
        font-weight: 400;
    }
    
    .hero-brand {
        font-size: 0.85rem;
        opacity: 0.5;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    /* Eligibility Result Colors */
    .status-eligible { color: #10b981 !important; font-weight: 700; }
    .status-partial { color: #f59e0b !important; font-weight: 700; }
    .status-not-eligible { color: #ef4444 !important; font-weight: 700; }
    
    .result-border-eligible { border-left: 6px solid #10b981; }
    .result-border-partial { border-left: 6px solid #f59e0b; }
    .result-border-not-eligible { border-left: 6px solid #ef4444; }

    /* Chat bubbles */
    .stChatMessage {
        background: var(--secondary-background-color);
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        margin-bottom: 15px;
    }
    
    /* Hide sidebar toggle */
    [data-testid="collapsedControl"] {
        display: none;
    }
</style>
"""

st.markdown(universal_css, unsafe_allow_html=True)

# --- RAG INITIALIZATION ---
@st.cache_resource
def get_retriever():
    vs = load_vector_store()
    if vs is None:
        return None
    return vs.as_retriever(search_kwargs={"k": 6})

retriever = get_retriever()

# --- HELPER FUNCTIONS ---
def handle_file_upload(uploaded_files):
    if not os.path.exists("data"):
        os.makedirs("data")
    count = 0
    for file in uploaded_files:
        file_path = os.path.join("data", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        count += 1
    return count

def check_eligibility(profile_data):
    if not retriever:
        return None
    # Generic query to pull all scholarship documents
    docs = retriever.invoke("scholarships requirements eligibility deadline criteria")
    context = "\\n\\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    You are an AI evaluating scholarship eligibility.
    Based strictly on the following context documents, evaluate the student's eligibility for EACH scholarship mentioned.
    
    STUDENT PROFILE:
    - Major/Course: {profile_data['major']}
    - CGPA: {profile_data['gpa']}
    - Family Income: ${profile_data['income']}
    - Year: {profile_data['year']}
    - Category: {profile_data['category']}
    
    CONTEXT DOCUMENTS:
    {context}
    
    Output a strictly valid JSON array of objects. Do NOT wrap it in markdown code blocks. Just the raw JSON.
    Format:
    [
      {{
        "name": "Scholarship Name",
        "status": "Eligible" | "Partial" | "Not Eligible",
        "reason": "Short explanation based on criteria matching.",
        "deadline": "Deadline Date or 'Not specified'"
      }}
    ]
    """
    
    models_to_try = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    response = None
    last_error = None
    
    for model_name in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
            response = llm.invoke(prompt)
            break  # Success!
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            last_error = e
            time.sleep(1)
            
    if not response:
        return [{
            "name": "System Currently Unavailable", 
            "status": "Not Eligible", 
            "reason": "All AI models are currently busy. Please try again in a few moments.", 
            "deadline": "N/A"
        }]
        
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception as e:
        print(f"Failed to parse JSON: {e}\\nContent: {response.content}")
        return []

# ==========================================
# 0. ADMIN AUTHENTICATION
# ==========================================
col_empty, col_admin = st.columns([8, 2])
with col_admin:
    if st.session_state.admin_authenticated:
        if st.button("Logout Admin"):
            st.session_state.admin_authenticated = False
            st.rerun()
    else:
        with st.popover("⚙️ Admin Login"):
            st.write("Enter credentials to access admin tools.")
            admin_user = st.text_input("Username")
            admin_pass = st.text_input("Password", type="password")
            if st.button("Login"):
                if admin_user == "admin" and admin_pass == "admin123":
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

# ==========================================
# 1. HEADER / HERO SECTION
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Find Scholarships You Actually Qualify For</div>
    <div class="hero-subtitle">Enter your details and instantly see matching scholarships</div>
    <div class="hero-brand">Powered by ScholarPath AI</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. PROFILE INPUT SECTION
# ==========================================
st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
with st.form("profile_form"):
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        course_opts = ["Select Course", "Computer Science", "Information Technology", "Electronics and Communication", "Mechanical Engineering", "Civil Engineering", "Electrical Engineering", "Business Administration", "Medicine", "Law", "Arts & Humanities", "Other"]
        current_course = st.session_state.profile.get("major", "Select Course")
        if not current_course: current_course = "Select Course"
        major = st.selectbox("Course / Major", course_opts, index=course_opts.index(current_course) if current_course in course_opts else 0)
        gpa = st.number_input("CGPA (out of 10.0)", min_value=0.0, max_value=10.0, value=float(st.session_state.profile["gpa"]), step=0.1)
    with p_col2:
        income = st.number_input("Annual Family Income ($)", min_value=0, value=int(st.session_state.profile["income"]), step=5000)
        year_opts = ["1st Year", "2nd Year", "3rd Year", "4th Year", "Postgraduate"]
        year = st.selectbox("Academic Year", year_opts, index=year_opts.index(st.session_state.profile["year"]) if st.session_state.profile["year"] in year_opts else 0)
    with p_col3:
        cat_opts = ["Select Category", "General", "OBC", "SC", "ST", "Minority", "Other"]
        current_cat = st.session_state.profile.get("category", "Select Category")
        category = st.selectbox("Category", cat_opts, index=cat_opts.index(current_cat) if current_cat in cat_opts else 0)
        
        st.write("") # spacing
        st.write("") # spacing
        
    col_submit, col_reset, _ = st.columns([3, 2, 5])
    with col_submit:
        submitted = st.form_submit_button("🎯 Check My Eligibility", type="primary")
    with col_reset:
        reset = st.form_submit_button("🔄 Clear Details")
        
    if reset:
        st.session_state.profile = {
            "major": "", "gpa": 8.0, "income": 50000, 
            "year": "1st Year", "category": "Select Category", "language": st.session_state.profile.get("language", "English")
        }
        st.session_state.eligibility_results = None
        st.rerun()

    if submitted:
        st.session_state.profile.update({
            "major": major, "gpa": gpa, "income": income, 
            "year": year, "category": category
        })
        
        with st.spinner("Analyzing scholarships against your profile..."):
            results = check_eligibility(st.session_state.profile)
            if results is not None:
                st.session_state.eligibility_results = results
            else:
                st.error("Failed to generate results or no documents found.")
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 3. RESULTS SECTION
# ==========================================
if st.session_state.eligibility_results is not None:
    st.markdown("### Your Scholarship Matches")
    
    if len(st.session_state.eligibility_results) == 0:
        st.info("No scholarships found to evaluate.")
    else:
        # Sort results: Eligible first, then Partial, then Not Eligible
        sorted_results = sorted(
            st.session_state.eligibility_results, 
            key=lambda x: 0 if x.get("status") == "Eligible" else (1 if x.get("status") == "Partial" else 2)
        )
        
        for item in sorted_results:
            status = item.get("status", "Not Eligible")
            status_class = "status-not-eligible"
            border_class = "result-border-not-eligible"
            icon = "❌"
            
            if status == "Eligible":
                status_class = "status-eligible"
                border_class = "result-border-eligible"
                icon = "✅"
            elif status == "Partial":
                status_class = "status-partial"
                border_class = "result-border-partial"
                icon = "⚠️"
                
            st.markdown(f"""
            <div class="result-card {border_class}">
                <h4 style="margin-top:0; margin-bottom: 8px;">{item.get('name', 'Unknown Scholarship')}</h4>
                <div style="margin-bottom: 12px; font-size: 1.05em;">
                    <span class="{status_class}">{icon} {status}</span> 
                    <span style="opacity: 0.6; margin: 0 10px;">|</span> 
                    <b>Deadline:</b> {item.get('deadline', 'N/A')}
                </div>
                <p style="margin:0; opacity: 0.9; line-height: 1.5;">{item.get('reason', '')}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("ℹ️ Fill your details and click **Check My Eligibility** to see matching scholarships here.")


st.write("")
st.write("")
st.divider()

# ==========================================
# 4. AI CHAT ASSISTANT
# ==========================================
st.markdown("### Ask Questions")
st.write("Have questions about required documents or the application process? Ask our assistant below.")

chat_container = st.container()

with chat_container:
    col1, col2 = st.columns([3, 1])
    with col2:
        st.session_state.profile["language"] = st.selectbox(
            "Reply Language", 
            ["English", "Telugu"], 
            index=["English", "Telugu"].index(st.session_state.profile.get("language", "English")),
            key="chat_lang"
        )
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me any questions about the scholarships, deadlines, or required documents."}]
            st.session_state.chat_history = []
            st.rerun()

    if retriever is None:
        st.warning("⚠️ No scholarships found in database.")
    elif not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ Google Gemini API Key is missing.")
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

        system_prompt = (
            "You are a helpful scholarship assistant.\n"
            "Use the provided context to answer the student's question accurately.\n"
            "If the answer is not in the context, state that you do not know.\n"
            f"CRITICAL: You MUST answer strictly in the following language: {st.session_state.profile['language']}.\n\n"
            "--- STUDENT PROFILE ---\n"
            f"- Major/Course: {st.session_state.profile['major']}\n"
            f"- CGPA: {st.session_state.profile['gpa']}\n"
            f"- Family Income: ${st.session_state.profile['income']}\n"
            f"- Year of Study: {st.session_state.profile['year']}\n"
            f"- Category: {st.session_state.profile['category']}\n"
            "-----------------------\n\n"
            "Context:\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        msg_container = st.container(height=400)
        
        with msg_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if user_input := st.chat_input("E.g., What documents do I need to prepare?"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with msg_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Checking policies..."):
                        try:
                            response = rag_chain.invoke({
                                "input": user_input,
                                "chat_history": st.session_state.chat_history
                            })
                            answer = response["answer"]
                        except Exception as e:
                            answer = "⚠️ **Server Busy:** The AI model is currently experiencing high demand. Please try again in a few moments."
                            print(f"Chat API Error: {e}")
                            
                        st.markdown(answer)
                        
                        st.session_state.chat_history.extend([
                            HumanMessage(content=user_input),
                            AIMessage(content=answer)
                        ])
                        st.session_state.messages.append({"role": "assistant", "content": answer})

st.write("")
st.write("")
st.divider()

# ==========================================
# 5. ADMIN SECTION (PROTECTED)
# ==========================================
if st.session_state.admin_authenticated:
    st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Manage Scholarships (Admin Dashboard)")
    st.write("Upload new policy documents or manage the existing database.")
    
    uploaded_files = st.file_uploader("Upload Documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    
    if st.button("Save Files to Database"):
        if uploaded_files:
            count = handle_file_upload(uploaded_files)
            st.success(f"Successfully saved {count} files!")
        else:
            st.warning("Please upload at least one file.")
            
    st.write("---")
    
    if st.button("🚀 Re-index Database", type="primary"):
        with st.spinner("Processing documents and building index..."):
            build_vector_store()
        st.success("Database successfully synchronized!")
        st.cache_resource.clear()
        
    st.write("---")
    
    if st.button("🗑️ Clear All Data"):
        if os.path.exists("data"):
            shutil.rmtree("data")
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
        st.warning("All documents and indexes have been wiped out.")
        st.cache_resource.clear()
        
    st.markdown("</div>", unsafe_allow_html=True)
