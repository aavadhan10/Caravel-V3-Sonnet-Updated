import streamlit as st
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def get_claude_response(query, availability_data, bios_data):
    """Get Claude's analysis of the best lawyer matches"""
    
    prompt = f"""Given these two datasets about lawyers and their availability, please analyze and recommend 3-5 best matches for the client's needs.

Client Need: {query}

Lawyer Availability Data:
{availability_data.to_string()}

Lawyer Bio Data:
{bios_data.to_string()}

Please recommend 3-5 lawyers who would be the best fit, considering their expertise and current availability. Format your response with:
1. A brief analysis of why these lawyers would be good matches
2. The specific recommendations with their key qualifications and availability
3. Any important notes about their expertise or availability"""

    try:
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None

def main():
    st.title("Caravel Law Lawyer Matcher")
    
    try:
        # Load the data files
        availability_data = pd.read_csv('Caravel Law Availability - October 18th, 2024.csv')
        bios_data = pd.read_csv('BD_Caravel.csv')
    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        return

    # Example queries
    st.write("### How can we help you find the right lawyer?")
    examples = [
        "I need a lawyer with trademark and IP experience who can start work soon",
        "Looking for someone experienced in employment law to help with HR policies",
        "Need a lawyer experienced with startups and financing",
        "Who would be best for drafting and negotiating SaaS agreements?"
    ]
    
    # Example query buttons
    col1, col2 = st.columns(2)
    for i, example in enumerate(examples):
        if i % 2 == 0:
            if col1.button(f"🔍 {example}"):
                st.session_state.query = example
                st.rerun()
        else:
            if col2.button(f"🔍 {example}"):
                st.session_state.query = example
                st.rerun()

    # Custom query input
    query = st.text_area(
        "Or describe what you're looking for:",
        value=st.session_state.get('query', ''),
        placeholder="Describe your legal needs..."
    )

    # Search and Clear buttons
    col1, col2 = st.columns([1, 4])
    search = col1.button("Find Lawyers")
    clear = col2.button("Clear")

    if clear:
        st.session_state.query = ''
        st.rerun()

    if search:
        st.session_state.query = query
        st.rerun()

    # Process search
    if st.session_state.get('query'):
        with st.spinner("Finding the best matches..."):
            results = get_claude_response(st.session_state.query, availability_data, bios_data)
            
        if results:
            st.write(results)

if __name__ == "__main__":
    main()
