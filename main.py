import streamlit as st
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

[Previous load_data() and format_lawyer_info() functions remain the same...]

def main():
    st.title("Caravel Law Lawyer Matcher")
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        st.error("Please set the ANTHROPIC_API_KEY environment variable")
        return
    
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Initialize session state if needed
    if 'query' not in st.session_state:
        st.session_state.query = ''

    # Example queries
    st.write("### How can we help you find the right lawyer?")
    st.write("Try asking something like:")
    examples = [
        "I need a lawyer with trademark experience who can start work soon",
        "Looking for someone with employment law experience to help with HR policies and contracts",
        "Need a lawyer experienced with startups and financing",
        "Who would be best for drafting and negotiating SaaS agreements?"
    ]
    
    # Create two columns for example queries
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
        value=st.session_state.query,
        help="Be as specific as you can about your needs"
    )

    col1, col2 = st.columns([1, 4])
    search = col1.button("Find Lawyers")
    clear = col2.button("Clear and Start Over")

    if clear:
        st.session_state.query = ''
        st.rerun()

    if search:
        st.session_state.query = query
        st.rerun()

    if st.session_state.query:
        with st.spinner("Analyzing available lawyers..."):
            analysis = get_claude_analysis(st.session_state.query, df)
        
        # Display the analysis in a clean format
        sections = analysis.split('\n\n')
        for section in sections:
            if section.startswith('ANALYSIS:'):
                with st.expander("Detailed Analysis", expanded=True):
                    st.write(section.replace('ANALYSIS:', '').strip())
            elif section.startswith('TOP RECOMMENDATIONS:'):
                st.write("### Top Recommendations")
                st.write(section.replace('TOP RECOMMENDATIONS:', '').strip())
            elif section.startswith('ALTERNATIVE CONSIDERATIONS:'):
                with st.expander("Alternative Options"):
                    st.write(section.replace('ALTERNATIVE CONSIDERATIONS:', '').strip())
            elif section.startswith('AVAILABILITY NOTES:'):
                with st.expander("Availability Information"):
                    st.write(section.replace('AVAILABILITY NOTES:', '').strip())
    elif not st.session_state.query and not clear:
        st.info("Please enter your requirements or select an example query.")

if __name__ == "__main__":
    main()
