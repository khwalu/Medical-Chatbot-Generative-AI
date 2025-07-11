import gradio as gr
import os
from typing import List, Tuple
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Import your RAG model components
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone.grpc import PineconeGRPC as Pinecone

# Global variables
rag_chain = None
model_loaded = False
chat_count = 0

def validate_environment():
    """Validate required environment variables"""
    required_vars = ['PINECONE_API_KEY', 'GEMINI_API_KEY']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        return False, f"❌ Missing environment variables: {', '.join(missing_vars)}"
    return True, "✅ Environment variables validated"

def initialize_rag_model():
    """Initialize the RAG model"""
    global rag_chain, model_loaded
    
    try:
        # Validate environment
        is_valid, message = validate_environment()
        if not is_valid:
            return False, message
            
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        index_name = "medicalbot"
        
        # Load existing vector store
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        # Setup retriever
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
        
        # Setup chat model
        chatModel = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=os.environ["GEMINI_API_KEY"],
            temperature=0.1,
            max_output_tokens=1000
        )
        
        # Create enhanced prompt template
        system_prompt = (
            "You are MedBot AI, a knowledgeable medical assistant. Use the following retrieved context to answer "
            "medical questions accurately and helpfully. If you don't know the answer, say so clearly. "
            "Provide clear, concise answers with relevant medical information. "
            "Always remind users to consult healthcare professionals for serious medical concerns.\n\n"
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create RAG chain
        question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        model_loaded = True
        return True, "✅ MedBot AI initialized successfully! Ready to help with medical questions."
        
    except Exception as e:
        model_loaded = False
        return False, f"❌ Error initializing model: {str(e)}"

def chat_response(message: str, history: List[Tuple[str, str]]) -> str:
    """Generate response using RAG model"""
    global rag_chain, model_loaded, chat_count
    
    if not model_loaded or rag_chain is None:
        return "⚠️ MedBot AI is not initialized. Please click 'Initialize MedBot' first."
    
    if not message.strip():
        return "Please ask a medical question."
    
    try:
        # Update chat count
        chat_count += 1
        
        # Get response from RAG chain
        response = rag_chain.invoke({"input": message})
        answer = response["answer"]
        
        # Add medical disclaimer for certain topics
        sensitive_topics = ['diagnosis', 'treatment', 'medication', 'dosage', 'emergency']
        if any(topic in message.lower() for topic in sensitive_topics):
            answer += "\n\n⚠️ **Important:** This information is for educational purposes only. Always consult with qualified healthcare professionals for medical advice, diagnosis, or treatment."
        
        return answer
        
    except Exception as e:
        return f"❌ Error processing your question: {str(e)}\n\nPlease try again or rephrase your question."

def initialize_model_interface():
    """Interface function to initialize the model"""
    success, message = initialize_rag_model()
    return message

def get_chat_stats():
    """Get current chat statistics"""
    global chat_count, model_loaded
    status = "🟢 Active" if model_loaded else "🔴 Not Initialized"
    return f"Status: {status} | Questions Asked: {chat_count}"

def clear_chat_history():
    """Clear chat history"""
    global chat_count
    chat_count = 0
    return [], "Chat history cleared!"

# Sample medical questions for quick testing
SAMPLE_QUESTIONS = [
    "What is diabetes and what are its symptoms?",
    "How can I prevent heart disease?",
    "What are the common causes of high blood pressure?",
    "What is the difference between Type 1 and Type 2 diabetes?",
    "How does asthma affect breathing?",
    "What are the warning signs of a stroke?",
    "How can I maintain good cardiovascular health?",
    "What causes migraine headaches?"
]

# Custom CSS for medical theme
css = """
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.gr-interface {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.gr-button {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 25px !important;
    color: white !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
}

.gr-chatbot {
    background: #f8f9fa !important;
    border-radius: 15px !important;
    border: 1px solid #e9ecef !important;
}

.gr-textbox {
    border-radius: 25px !important;
    border: 2px solid #667eea !important;
}

.gr-textbox:focus {
    border-color: #764ba2 !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

.medical-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    border-radius: 15px;
    margin-bottom: 20px;
}

.stats-display {
    background: linear-gradient(45deg, #43e97b, #38f9d7);
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
    color: #333;
    margin: 10px 0;
}
"""

# Create the main interface
with gr.Blocks(css=css, title="🩺 MedBot AI - Medical Assistant", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown("""
    <div class="medical-header">
        <h1>🩺 MedBot AI</h1>
        <h3>Your Intelligent Medical Assistant</h3>
        <p>Advanced RAG-powered medical knowledge at your fingertips</p>
    </div>
    """)
    
    with gr.Row():
        # Main chat interface
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(
                value=[],
                height=500,
                placeholder="MedBot AI will appear here once initialized...",
                show_label=False,
                avatar_images=("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", 
                              "https://cdn-icons-png.flaticon.com/512/2138/2138440.png"),
                bubble_full_width=False
            )
            
            # Chat input
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask your medical question here... (e.g., 'What are the symptoms of diabetes?')",
                    show_label=False,
                    scale=4,
                    container=False
                )
                send_btn = gr.Button("Send 📤", scale=1, variant="primary")
            
            # Sample questions
            gr.Markdown("### 💡 Quick Questions:")
            with gr.Row():
                for i in range(0, len(SAMPLE_QUESTIONS), 2):
                    with gr.Column():
                        if i < len(SAMPLE_QUESTIONS):
                            sample_btn1 = gr.Button(f"💬 {SAMPLE_QUESTIONS[i][:30]}...", size="sm")
                            sample_btn1.click(
                                lambda x=SAMPLE_QUESTIONS[i]: x, 
                                outputs=msg
                            )
                        if i + 1 < len(SAMPLE_QUESTIONS):
                            sample_btn2 = gr.Button(f"💬 {SAMPLE_QUESTIONS[i+1][:30]}...", size="sm")
                            sample_btn2.click(
                                lambda x=SAMPLE_QUESTIONS[i+1]: x, 
                                outputs=msg
                            )
        
        # Sidebar
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Control Panel")
            
            # Model initialization
            init_btn = gr.Button("🚀 Initialize MedBot", variant="primary", size="lg")
            init_status = gr.Textbox(
                value="Click 'Initialize MedBot' to start",
                label="Status",
                interactive=False
            )
            
            # Statistics
            gr.Markdown("### 📊 Statistics")
            stats_display = gr.Textbox(
                value="Status: 🔴 Not Initialized | Questions Asked: 0",
                label="Chat Stats",
                interactive=False
            )
            
            # Controls
            gr.Markdown("### 🎛️ Controls")
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
            
            # Quick info
            gr.Markdown("""
            ### ℹ️ Quick Info
            - **Purpose**: Medical information assistant
            - **Data**: Trained on medical literature
            - **Disclaimer**: Not a replacement for professional medical advice
            - **Usage**: Ask clear, specific medical questions
            """)
    
    # Medical disclaimer
    gr.Markdown("""
    <div style="background: linear-gradient(45deg, #ffd89b, #19547b); padding: 15px; border-radius: 10px; margin: 20px 0; color: white; text-align: center;">
        <strong>⚠️ IMPORTANT MEDICAL DISCLAIMER</strong><br>
        This AI provides general medical information only. Always consult qualified healthcare professionals 
        for medical advice, diagnosis, or treatment. In case of emergency, contact emergency services immediately.
    </div>
    """)
    
    # Event handlers
    def respond(message, chat_history):
        if message.strip():
            bot_message = chat_response(message, chat_history)
            chat_history.append((message, bot_message))
        return "", chat_history
    
    def update_stats():
        return get_chat_stats()
    
    def clear_and_update():
        new_history, message = clear_chat_history()
        return new_history, message, get_chat_stats()
    
    # Connect events
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send_btn.click(respond, [msg, chatbot], [msg, chatbot])
    
    init_btn.click(initialize_model_interface, outputs=init_status)
    
    clear_btn.click(clear_and_update, outputs=[chatbot, init_status, stats_display])
    refresh_btn.click(update_stats, outputs=stats_display)
    
    # Auto-refresh stats every few interactions
    chatbot.change(update_stats, outputs=stats_display)

# Launch configuration
if __name__ == "__main__":
    # Initialize model on startup (optional)
    print("🩺 Starting MedBot AI...")
    print("🔧 Environment check...")
    is_valid, message = validate_environment()
    print(message)
    
    if is_valid:
        print("🚀 Auto-initializing MedBot AI...")
        success, init_message = initialize_rag_model()
        print(init_message)
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public URL
        inbrowser=True,         # Open in browser automatically
        show_error=True,        # Show detailed errors
        favicon_path=None,      # You can add a custom favicon
        show_api=True,          # Show API documentation
        max_threads=10          # Max concurrent users
    )