# client.py
import streamlit as st
import requests
from datetime import datetime
import re
import time

API = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="🔄 Hybrid Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        word-wrap: break-word;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.bot {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
    }
    .chat-message .header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
        color: #666666;
    }
    .stButton>button {
        width: 100%;
    }
    .mode-info {
        font-size: 0.9rem;
        color: #666666;
        margin-top: 0.5rem;
    }
    .message-content {
        white-space: pre-wrap;
    }
    .link-text {
        color: #0969da;
        text-decoration: none;
    }
    .link-text:hover {
        text-decoration: underline;
    }
    .emoji-section {
        font-weight: bold;
        color: #444;
    }
    .reasoning {
        background-color: #f8f9fa;
        border-left: 3px solid #5c6bc0;
        padding: 10px;
        margin: 10px 0;
    }
    .sources {
        background-color: #f1f8e9;
        border-left: 3px solid #8bc34a;
        padding: 10px;
        margin: 10px 0;
    }
    .llm-response {
        background-color: #e3f2fd;
        border-left: 3px solid #2196f3;
        padding: 10px;
        margin: 10px 0;
    }
    .web-response {
        background-color: #fff8e1;
        border-left: 3px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }
    .thinking-step {
        background-color: #f5f5f5;
        border-left: 3px solid #9e9e9e;
        padding: 10px;
        margin: 5px 0;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }
    .thinking-title {
        font-weight: bold;
        margin-bottom: 5px;
        color: #424242;
    }
    .thinking-content {
        color: #616161;
    }
    .chain-of-thought {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .chain-header {
        font-weight: bold;
        margin-bottom: 10px;
        color: #424242;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 5px;
    }
    .step-number {
        display: inline-block;
        background-color: #5c6bc0;
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 12px;
        text-align: center;
        margin-right: 8px;
    }
    .status-positive {
        color: #4caf50;
        font-weight: bold;
    }
    .status-negative {
        color: #f44336;
        font-weight: bold;
    }
    .web-search-result {
        border-left: 3px solid #4caf50;
        padding-left: 10px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.title("🔄 Settings")
    
    # Mode selection with tooltip
    mode = st.radio(
        "Select Mode",
        ["LLM only", "Search only", "Hybrid"],
        help="LLM only: Uses only the language model\n"
             "Search only: Searches the web and summarizes\n"
             "Hybrid: Intelligently combines web search with LLM when needed"
    )
    
    # Mode descriptions
    mode_descriptions = {
        "LLM only": "Quick responses using the model's knowledge",
        "Search only": "Web search results with sources",
        "Hybrid": "Combines web search with AI for best answers"
    }
    st.markdown(f'<div class="mode-info">{mode_descriptions[mode]}</div>', unsafe_allow_html=True)
    
    # Show reasoning process
    if mode == "Hybrid":
        show_reasoning = st.checkbox("Show detailed reasoning process", value=True, 
                                    help="Display the step-by-step reasoning process in hybrid mode")
    
    # Clear history button
    if st.button("🗑️ Clear Chat History", help="Clear all chat history"):
        if st.session_state.history:
            st.session_state.history = []
            st.success("Chat history cleared!")
            st.rerun()

# Main content
st.title("🔄 Hybrid Chatbot")

def extract_sections(text):
    """Extract different sections from a hybrid response."""
    sections = {
        "llm_response": "",
        "reasoning": "",
        "web_summary": "",
        "augmented_response": "",
        "sources": []
    }
    
    # Find LLM Response
    llm_match = re.search(r'🤖 \[.*LLM Response\]\s*\n\n(.*?)(?=\n\n[⚠️📊]|\n\n📚|$)', text, re.DOTALL)
    if llm_match:
        sections["llm_response"] = llm_match.group(1).strip()
    
    # Find Reasoning
    reason_match = re.search(r'⚠️ Reason for web augmentation:(.*?)(?=\n\n🌐|\n\n📚|$)', text, re.DOTALL)
    if reason_match:
        sections["reasoning"] = reason_match.group(1).strip()
    
    # Find Web Summary
    web_match = re.search(r'🌐 \[Web Summary\]\s*\n(.*?)(?=\n\n🔄|\n\n📚|$)', text, re.DOTALL)
    if web_match:
        sections["web_summary"] = web_match.group(1).strip()
    
    # Find Augmented Response
    aug_match = re.search(r'🔄 \[Web-Augmented Response\]\s*\n\n(.*?)(?=\n\n📚|$)', text, re.DOTALL)
    if aug_match:
        sections["augmented_response"] = aug_match.group(1).strip()
    
    # Find Sources
    sources_match = re.search(r'📚 Sources:\s*\n(.*?)$', text, re.DOTALL)
    if sources_match:
        sources_text = sources_match.group(1)
        source_entries = re.findall(r'(\d+)\. (.*?)\n\s*(https?://\S+)', sources_text)
        for _, title, link in source_entries:
            sections["sources"].append({"title": title.strip(), "link": link.strip()})
    
    return sections

def stream_response(resp):
    """Read streaming text and yield to UI."""
    text = ""
    for chunk in resp.iter_content(chunk_size=16, decode_unicode=True):
        if chunk:
            text += chunk
            yield text

def format_streamed_response(text, mode):
    """Format streamed response based on mode."""
    if mode == "LLM only":
        # Remove the disclaimer line if present
        if text.startswith("Note: This response is based on"):
            parts = text.split("\n\n", 1)
            if len(parts) > 1:
                formatted = f"<div class='disclaimer'>{parts[0]}</div>\n\n{parts[1]}"
            else:
                formatted = text
        else:
            formatted = text
        return formatted
    
    elif mode == "Search only":
        if text.startswith("📝 Summary:"):
            # Already formatted correctly
            return text
        else:
            return text
    
    elif mode == "Hybrid":
        # For hybrid, we'll format in the send_query function using tabs
        return text

def send_query(prompt: str):
    # Add user message to history
    st.session_state.history.insert(0, {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now(),
        "mode": mode
    })
    
    # Create placeholder for bot response
    placeholder = st.empty()
    
    # Send request based on mode
    try:
        if mode == "LLM only":
            resp = requests.post(
                f"{API}/chat/llm_only",
                json={"prompt": prompt},
                stream=True
            )
            
            # Stream the response
            full_response = ""
            with placeholder.container():
                for partial in stream_response(resp):
                    full_response = partial
                    formatted = format_streamed_response(partial, mode)
                    placeholder.markdown(formatted, unsafe_allow_html=True)
                    
        elif mode == "Search only":
            resp = requests.get(
                f"{API}/search",
                params={"q": prompt},
                stream=True
            )
            
            # Stream the response
            full_response = ""
            with placeholder.container():
                for partial in stream_response(resp):
                    full_response = partial
                    formatted = format_streamed_response(partial, mode)
                    placeholder.markdown(formatted, unsafe_allow_html=True)
                    
        else:  # hybrid
            resp = requests.post(
                f"{API}/chat/hybrid",
                json={"prompt": prompt},
                stream=True
            )
            
            # Stream the response with chain of thought
            full_response = ""
            chain_of_thought_data = {
                "llm_response": {"title": "Initial LLM Response", "content": "", "complete": False},
                "need_web": {"title": "Web Search Needed?", "content": "", "complete": False},
                "reason": {"title": "Reason", "content": "", "complete": False},
                "web_summary": {"title": "Web Search Summary", "content": "", "complete": False},
                "augmented_response": {"title": "Final Response", "content": "", "complete": False}
            }
            
            with placeholder.container():
                # Create placeholders for streaming and reasoning
                response_placeholder = st.empty()
                
                if 'show_reasoning' in locals() and show_reasoning:
                    reasoning_placeholder = st.empty()
                    
                    # Start with initial "thinking" state
                    reasoning_html = """
                    <div class="chain-of-thought">
                        <div class="chain-header">🧠 Chain of Thought</div>
                        <div class="thinking-step">
                            <div class="thinking-title"><span class="step-number">1</span> Generating LLM response...</div>
                            <div class="thinking-content">Waiting...</div>
                        </div>
                    </div>
                    """
                    reasoning_placeholder.markdown(reasoning_html, unsafe_allow_html=True)
                
                # Stream and process the response
                for partial in stream_response(resp):
                    full_response = partial
                    
                    # Update the response display
                    response_placeholder.markdown(partial)
                    
                    # Extract and update chain of thought data
                    if 'show_reasoning' in locals() and show_reasoning:
                        # Check for LLM response
                        llm_match = re.search(r'🤖 \[.*LLM Response\]\s*\n\n(.*?)(?=\n\n[⚠️📊🌐]|\n\n📚|$)', partial, re.DOTALL)
                        if llm_match and not chain_of_thought_data["llm_response"]["complete"]:
                            chain_of_thought_data["llm_response"]["content"] = llm_match.group(1).strip()
                            chain_of_thought_data["llm_response"]["complete"] = True
                            
                            # Update need_web based on response continuation
                            if "⚠️ Reason for web augmentation:" in partial:
                                chain_of_thought_data["need_web"]["content"] = "<span class='status-positive'>Yes</span>"
                                chain_of_thought_data["need_web"]["complete"] = True
                            elif "📌 Note: This response is based on the model's training data" in partial:
                                chain_of_thought_data["need_web"]["content"] = "<span class='status-negative'>No</span>"
                                chain_of_thought_data["need_web"]["complete"] = True
                        
                        # Check for reasoning
                        reason_match = re.search(r'⚠️ Reason for web augmentation:(.*?)(?=\n\n🌐|\n\n📚|$)', partial, re.DOTALL)
                        if reason_match and not chain_of_thought_data["reason"]["complete"]:
                            chain_of_thought_data["reason"]["content"] = reason_match.group(1).strip()
                            chain_of_thought_data["reason"]["complete"] = True
                        
                        # Check for web summary
                        web_match = re.search(r'🌐 \[Web Summary\]\s*\n(.*?)(?=\n\n🔄|\n\n📚|$)', partial, re.DOTALL)
                        if web_match and not chain_of_thought_data["web_summary"]["complete"]:
                            chain_of_thought_data["web_summary"]["content"] = web_match.group(1).strip()
                            chain_of_thought_data["web_summary"]["complete"] = True
                        
                        # Check for augmented response
                        aug_match = re.search(r'🔄 \[Web-Augmented Response\]\s*\n\n(.*?)(?=\n\n📚|$)', partial, re.DOTALL)
                        if aug_match and not chain_of_thought_data["augmented_response"]["complete"]:
                            chain_of_thought_data["augmented_response"]["content"] = aug_match.group(1).strip()
                            chain_of_thought_data["augmented_response"]["complete"] = True
                        
                        # Build reasoning HTML
                        reasoning_html = """
                        <div class="chain-of-thought">
                            <div class="chain-header">🧠 Chain of Thought</div>
                        """
                        
                        # Step 1: LLM Response
                        reasoning_html += f"""
                        <div class="thinking-step">
                            <div class="thinking-title"><span class="step-number">1</span> Initial LLM Response</div>
                            <div class="thinking-content">
                        """
                        if chain_of_thought_data["llm_response"]["complete"]:
                            reasoning_html += f"""
                            <div class="llm-response">{chain_of_thought_data["llm_response"]["content"]}</div>
                            """
                        else:
                            reasoning_html += "Generating..."
                        reasoning_html += "</div></div>"
                        
                        # Step 2: Web Search Needed?
                        reasoning_html += f"""
                        <div class="thinking-step">
                            <div class="thinking-title"><span class="step-number">2</span> Web Search Needed?</div>
                            <div class="thinking-content">
                        """
                        if chain_of_thought_data["need_web"]["complete"]:
                            reasoning_html += f"{chain_of_thought_data['need_web']['content']}"
                            
                            # Add reasoning if available
                            if chain_of_thought_data["reason"]["complete"]:
                                reasoning_html += f"""
                                <div class="reasoning">
                                    <strong>Reason:</strong> {chain_of_thought_data["reason"]["content"]}
                                </div>
                                """
                        else:
                            reasoning_html += "Evaluating..."
                        reasoning_html += "</div></div>"
                        
                        # Step 3: Web Search Results (if needed)
                        if "Yes" in chain_of_thought_data["need_web"]["content"]:
                            reasoning_html += f"""
                            <div class="thinking-step">
                                <div class="thinking-title"><span class="step-number">3</span> Web Search Results</div>
                                <div class="thinking-content">
                            """
                            if chain_of_thought_data["web_summary"]["complete"]:
                                reasoning_html += f"""
                                <div class="web-search-result">
                                    {chain_of_thought_data["web_summary"]["content"]}
                                </div>
                                """
                            else:
                                reasoning_html += "Searching web..."
                            reasoning_html += "</div></div>"
                            
                            # Step 4: Augmented Response
                            reasoning_html += f"""
                            <div class="thinking-step">
                                <div class="thinking-title"><span class="step-number">4</span> Final Augmented Response</div>
                                <div class="thinking-content">
                            """
                            if chain_of_thought_data["augmented_response"]["complete"]:
                                reasoning_html += f"""
                                <div class="web-response">
                                    {chain_of_thought_data["augmented_response"]["content"]}
                                </div>
                                """
                            else:
                                reasoning_html += "Generating augmented response..."
                            reasoning_html += "</div></div>"
                        
                        reasoning_html += "</div>"  # Close chain-of-thought div
                        reasoning_placeholder.markdown(reasoning_html, unsafe_allow_html=True)
                
                # After streaming is complete, create tabs with sections if not showing reasoning
                if full_response and not ('show_reasoning' in locals() and show_reasoning):
                    sections = extract_sections(full_response)
                    
                    # Create tabs
                    tab_names = ["Complete Answer"]
                    if sections["reasoning"]:
                        tab_names.append("Reasoning")
                    if sections["llm_response"] and sections["augmented_response"]:
                        tab_names.append("Original vs Augmented")
                    if sections["sources"]:
                        tab_names.append("Sources")
                    
                    tabs = st.tabs(tab_names)
                    
                    # Complete Answer tab
                    with tabs[0]:
                        st.markdown(full_response)
                    
                    # Reasoning tab (if available)
                    if "Reasoning" in tab_names:
                        idx = tab_names.index("Reasoning")
                        with tabs[idx]:
                            st.markdown(f"### Why web search was used")
                            st.markdown(f"<div class='reasoning'>{sections['reasoning']}</div>", unsafe_allow_html=True)
                    
                    # Original vs Augmented tab (if both available)
                    if "Original vs Augmented" in tab_names:
                        idx = tab_names.index("Original vs Augmented")
                        with tabs[idx]:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("### LLM Response")
                                st.markdown(f"<div class='llm-response'>{sections['llm_response']}</div>", unsafe_allow_html=True)
                            with col2:
                                st.markdown("### Web-Augmented Response")
                                st.markdown(f"<div class='web-response'>{sections['augmented_response']}</div>", unsafe_allow_html=True)
                    
                    # Sources tab (if available)
                    if "Sources" in tab_names:
                        idx = tab_names.index("Sources")
                        with tabs[idx]:
                            st.markdown("### Web Sources")
                            for source in sections["sources"]:
                                st.markdown(f"[{source['title']}]({source['link']})")
        
        # Save response and chain of thought to history
        response_data = {
            "role": "bot",
            "content": full_response,
            "timestamp": datetime.now(),
            "mode": mode
        }
        
        # If hybrid and reasoning was shown, save the reasoning data
        if mode == "Hybrid" and 'show_reasoning' in locals() and show_reasoning and 'chain_of_thought_data' in locals():
            response_data["reasoning_data"] = chain_of_thought_data
        
        st.session_state.history.insert(0, response_data)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.history.insert(0, {
            "role": "bot",
            "content": f"Error: {str(e)}",
            "timestamp": datetime.now(),
            "mode": mode
        })

# Chat input
with st.container():
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Your question:",
            key="inp",
            placeholder="Type your question here...",
            help="Enter your question and press Send or hit Enter"
        )
    with col2:
        st.write("")  # For alignment
        if st.button("Send", help="Send your question to the chatbot"):
            if query:
                send_query(query)
                st.rerun()

# Chat history
st.write("---")
for msg in st.session_state.history:
    timestamp = msg["timestamp"].strftime("%H:%M:%S")
    chat_mode = msg.get("mode", "")
    
    if msg["role"] == "user":
        st.markdown(f"""
            <div class="chat-message user">
                <div class="header">
                    <span>You</span>
                    <span>{timestamp}</span>
                </div>
                <div class="message-content">{msg["content"]}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Add mode badge if available
        mode_badge = f"<span style='background-color: #e0f7fa; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; margin-left: 8px;'>{chat_mode}</span>" if chat_mode else ""
        
        # Regular message content
        message_html = f"""
            <div class="chat-message bot">
                <div class="header">
                    <span>Bot{mode_badge}</span>
                    <span>{timestamp}</span>
                </div>
                <div class="message-content">{msg["content"]}</div>
        """
        
        # Add reasoning data if available (for hybrid mode)
        if "reasoning_data" in msg and chat_mode == "Hybrid":
            reasoning_data = msg["reasoning_data"]
            
            # Only display reasoning if data is complete
            if reasoning_data["llm_response"]["complete"]:
                message_html += """
                <div class="chain-of-thought" style="margin-top: 15px;">
                    <div class="chain-header">🧠 Reasoning Process</div>
                """
                
                # Step 1: LLM Response
                message_html += f"""
                <div class="thinking-step">
                    <div class="thinking-title"><span class="step-number">1</span> Initial LLM Response</div>
                    <div class="thinking-content">
                        <div class="llm-response">{reasoning_data["llm_response"]["content"]}</div>
                    </div>
                </div>
                """
                
                # Step 2: Web Search Needed?
                message_html += f"""
                <div class="thinking-step">
                    <div class="thinking-title"><span class="step-number">2</span> Web Search Needed?</div>
                    <div class="thinking-content">
                        {reasoning_data["need_web"]["content"]}
                """
                
                # Add reasoning if available
                if reasoning_data["reason"]["complete"]:
                    message_html += f"""
                    <div class="reasoning">
                        <strong>Reason:</strong> {reasoning_data["reason"]["content"]}
                    </div>
                    """
                
                message_html += "</div></div>"
                
                # Step 3: Web Search Results (if needed)
                if "Yes" in reasoning_data["need_web"]["content"] and reasoning_data["web_summary"]["complete"]:
                    message_html += f"""
                    <div class="thinking-step">
                        <div class="thinking-title"><span class="step-number">3</span> Web Search Results</div>
                        <div class="thinking-content">
                            <div class="web-search-result">
                                {reasoning_data["web_summary"]["content"]}
                            </div>
                        </div>
                    </div>
                    """
                    
                    # Step 4: Augmented Response
                    if reasoning_data["augmented_response"]["complete"]:
                        message_html += f"""
                        <div class="thinking-step">
                            <div class="thinking-title"><span class="step-number">4</span> Final Augmented Response</div>
                            <div class="thinking-content">
                                <div class="web-response">
                                    {reasoning_data["augmented_response"]["content"]}
                                </div>
                            </div>
                        </div>
                        """
                
                message_html += "</div>"  # Close chain-of-thought div
            
        message_html += "</div>"  # Close chat-message div
        st.markdown(message_html, unsafe_allow_html=True)
