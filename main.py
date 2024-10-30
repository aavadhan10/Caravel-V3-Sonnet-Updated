import streamlit as st
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import time
import re

# Load environment variables
load_dotenv()

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def prepare_lawyer_summary(availability_data, bios_data):
    """Create a more detailed summary of lawyer information"""
    # Merge the dataframes
    availability_data['Name'] = availability_data['What is your name?'].str.strip()
    bios_data['Name'] = bios_data['First Name'].str.strip() + ' ' + bios_data['Last Name'].str.strip()
    
    lawyers_summary = []
    
    # Get available lawyers
    available_lawyers = availability_data[
        availability_data['Do you have capacity to take on new work?'].str.lower().isin(['yes', 'maybe'])
    ]
    
    for _, row in available_lawyers.iterrows():
        name = row['Name']
        bio_row = bios_data[bios_data['Name'] == name]
        
        if not bio_row.empty:
            bio_row = bio_row.iloc[0]
            summary = {
                'name': name,
                'level': bio_row.get('Level/Title', ''),
                'call_year': bio_row.get('Call', ''),
                'availability_status': row['Do you have capacity to take on new work?'],
                'practice_areas': bio_row.get('Area of Practise + Add Info', ''),
                'industry_experience': bio_row.get('Industry Experience', ''),
                'previous_companies': bio_row.get('Previous Companies/Firms', ''),
                'notable_experience': bio_row.get('Notable Items/Personal Details', ''),
                'days_available': row.get('What is your capacity to take on new work for the forseeable future? Days per week', ''),
                'hours_available': row.get('What is your capacity to take on new work for the foreseeable future? Hours per month', ''),
                'availability_notes': row.get('Do you have any comments or instructions you should let us know about that may impact your short/long-term availability? For instance, are you going on vacation (please provide exact dates)?', ''),
                'engagement_types': row.get('What type of engagement would you like to consider?', ''),
                'location': bio_row.get('Location', ''),
                'languages': bio_row.get('Languages', '')
            }
            lawyers_summary.append(summary)
    
    return lawyers_summary

def extract_recommended_lawyers(analysis_text, lawyers_summary):
    """Extract recommended lawyer names from Claude's analysis"""
    recommended_lawyers = []
    
    # Look for numbered recommendations
    pattern = r'\d+\.\s+([A-Za-z\s]+):'
    matches = re.findall(pattern, analysis_text)
    
    # Get full lawyer info for each recommendation
    for name in matches:
        name = name.strip()
        lawyer_info = next((lawyer for lawyer in lawyers_summary if lawyer['name'].lower() == name.lower()), None)
        if lawyer_info:
            recommended_lawyers.append(lawyer_info)
    
    return recommended_lawyers

def get_claude_response(query, lawyers_summary):
    """Get Claude's analysis of the best lawyer matches"""
    
    # Create a more detailed summary text
    summary_text = "Available Lawyers:\n"
    for lawyer in lawyers_summary:
        summary_text += f"\n{lawyer['name']} ({lawyer['level']}, Called to Bar: {lawyer['call_year']}):"
        summary_text += f"\n- Practice Areas: {lawyer['practice_areas']}"
        summary_text += f"\n- Industry Experience: {lawyer['industry_experience']}"
        if lawyer['previous_companies']:
            summary_text += f"\n- Previous Experience: {lawyer['previous_companies']}"
        if lawyer['notable_experience']:
            summary_text += f"\n- Notable Experience: {lawyer['notable_experience']}"
        summary_text += f"\n- Current Availability: {lawyer['availability_status']}"
        summary_text += f"\n  • Days/week: {lawyer['days_available']}"
        summary_text += f"\n  • Hours/month: {lawyer['hours_available']}"
        summary_text += f"\n  • Engagement Types: {lawyer['engagement_types']}"
        summary_text += f"\n  • Location: {lawyer['location']}\n"
    
    prompt = f"""You are helping match lawyers to client needs. Please analyze each lawyer's expertise and availability carefully and be direct about matches or lack thereof.

Client Need: {query}

Available Lawyers:
{summary_text}

If any lawyers match the client's needs, please recommend 3-5 best matches, clearly explaining:
1. Their relevant expertise and experience for this specific need
2. Their current availability
3. Why they would be a good fit

If no lawyers have demonstrated expertise in the required area, please say so directly and explain what expertise is missing.

Be specific in your recommendations and only suggest lawyers who have clear evidence of the required expertise in their profiles.

Format recommendations as numbered points starting with the lawyer's name, like:
1. [Lawyer Name]: [Explanation]
2. [Lawyer Name]: [Explanation]

Important: Only recommend lawyers if you see clear evidence in their profile that they have the specific expertise needed."""

    try:
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return response.content[0].text
    except Exception as e:
        if 'rate_limit_error' in str(e):
            st.warning("Rate limit reached. Waiting a moment before trying again...")
            time.sleep(5)
            try:
                response = anthropic.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                return response.content[0].text
            except Exception as e2:
                st.error("Still unable to get recommendations. Please try again in a moment.")
                return None
        st.error(f"Error getting recommendations: {str(e)}")
        return None

def main():
    st.title("Caravel Law Lawyer Matcher")
    
    try:
        # Load the data files with correct names
        availability_data = pd.read_csv('Caravel Law Availability - October 18th, 2024.csv')
        bios_data = pd.read_csv('BD_Caravel.csv')
        
        # Prepare summary once at startup
        lawyers_summary = prepare_lawyer_summary(availability_data, bios_data)
        
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
            analysis = get_claude_response(st.session_state.query, lawyers_summary)
            
        if analysis:
            # First display the analysis
            st.write("### Analysis")
            st.write(analysis)
            
            # Then extract and display availability details for recommended lawyers
            recommended_lawyers = extract_recommended_lawyers(analysis, lawyers_summary)
            
            if recommended_lawyers:
                st.write("\n### Detailed Availability Information")
                for lawyer in recommended_lawyers:
                    with st.expander(f"📅 {lawyer['name']}'s Availability"):
                        st.write("**Days per Week:**", lawyer['days_available'])
                        st.write("**Hours per Month:**", lawyer['hours_available'])
                        if lawyer['engagement_types']:
                            st.write("**Preferred Engagement Types:**", lawyer['engagement_types'])
                        if lawyer['availability_notes'] and lawyer['availability_notes'].lower() not in ['n/a', 'na', 'none', 'no']:
                            st.write("**Availability Notes:**", lawyer['availability_notes'])

if __name__ == "__main__":
    main()
