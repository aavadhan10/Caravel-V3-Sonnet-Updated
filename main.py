import streamlit as st
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def extract_recommended_lawyers(analysis_text, availability_df):
    """Extract recommended lawyer names and their availability info"""
    recommended_lawyers = []
    
    # Look for numbered recommendations
    pattern = r'\d+\.\s+([A-Za-z\s]+):'
    matches = re.findall(pattern, analysis_text)
    
    # Get availability info for each recommended lawyer
    for name in matches:
        name = name.strip()
        lawyer_info = availability_df[availability_df['What is your name?'].str.contains(name, case=False, na=False)]
        if not lawyer_info.empty:
            lawyer = {
                'name': name,
                'days_available': lawyer_info['What is your capacity to take on new work for the forseeable future? Days per week'].iloc[0],
                'hours_available': lawyer_info['What is your capacity to take on new work for the foreseeable future? Hours per month'].iloc[0],
                'engagement_types': lawyer_info['What type of engagement would you like to consider?'].iloc[0],
                'availability_notes': lawyer_info['Do you have any comments or instructions you should let us know about that may impact your short/long-term availability? For instance, are you going on vacation (please provide exact dates)?'].iloc[0]
            }
            recommended_lawyers.append(lawyer)
    
    return recommended_lawyers

def get_claude_response(query, availability_df, bios_df):
    """Get Claude's analysis - simple as if files were uploaded directly"""
    
    prompt = f"""I have two CSV files with lawyer information. Please look at both and recommend lawyers that match this need:

{query}

First file (Availability information):
{availability_df.to_string()}

Second file (Lawyer bios and expertise):
{bios_df.to_string()}

Please analyze these files and recommend any lawyers that clearly match the need. If no lawyers have the specific expertise required, please say so directly. Be honest - only recommend lawyers if you see clear evidence they have the relevant experience.

If recommending lawyers, please format their names clearly with numbers like:
1. [Lawyer Name]: [Explanation]
2. [Lawyer Name]: [Explanation]"""

    try:
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
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
        availability_df = pd.read_csv('Caravel Law Availability - October 18th, 2024.csv')
        bios_df = pd.read_csv('BD_Caravel.csv')
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
            if col1.button(f"🔍 {example}", key=f"example_{i}"):
                st.session_state.query = example
                st.rerun()
        else:
            if col2.button(f"🔍 {example}", key=f"example_{i}"):
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
            analysis = get_claude_response(st.session_state.query, availability_df, bios_df)
            
        if analysis:
            # Display Claude's analysis
            st.write(analysis)
            
            # Extract and show availability for recommended lawyers
            recommended_lawyers = extract_recommended_lawyers(analysis, availability_df)
            
            if recommended_lawyers:
                st.write("\n### Availability Details")
                for lawyer in recommended_lawyers:
                    with st.expander(f"📅 {lawyer['name']}'s Availability"):
                        st.write("**Days per Week:**", lawyer['days_available'])
                        st.write("**Hours per Month:**", lawyer['hours_available'])
                        if lawyer['engagement_types'] and str(lawyer['engagement_types']).lower() not in ['n/a', 'na', 'none', 'no']:
                            st.write("**Preferred Engagement Types:**", lawyer['engagement_types'])
                        if lawyer['availability_notes'] and str(lawyer['availability_notes']).lower() not in ['n/a', 'na', 'none', 'no']:
                            st.write("**Availability Notes:**", lawyer['availability_notes'])

if __name__ == "__main__":
    main()
