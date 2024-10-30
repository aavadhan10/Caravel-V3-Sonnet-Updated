import streamlit as st
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def load_data():
    """Load and merge the data files"""
    availability_df = pd.read_csv('Caravel Law Availability - October 18th, 2024.csv')
    bios_df = pd.read_csv('BD_Caravel.csv')
    
    # Clean up and merge data
    availability_df['Name'] = availability_df['What is your name?'].str.strip()
    bios_df['Name'] = bios_df['First Name'].str.strip() + ' ' + bios_df['Last Name'].str.strip()
    
    return pd.merge(availability_df, bios_df, on='Name', how='inner')

def format_lawyer_info(df):
    """Format lawyer information in a clear, readable way"""
    lawyers_info = []
    
    for _, row in df.iterrows():
        # Only include lawyers who are available
        if str(row.get('Do you have capacity to take on new work?', '')).lower() == 'yes':
            lawyer = {
                'name': row['Name'],
                'availability': {
                    'days_per_week': str(row.get('What is your capacity to take on new work for the forseeable future? Days per week', '')),
                    'hours_per_month': str(row.get('What is your capacity to take on new work for the foreseeable future? Hours per month', '')),
                    'notes': str(row.get('Do you have any comments or instructions you should let us know about that may impact your short/long-term availability? For instance, are you going on vacation (please provide exact dates)?', '')),
                    'engagement_types': str(row.get('What type of engagement would you like to consider?', ''))
                },
                'expertise': {
                    'level': str(row.get('Level/Title', '')),
                    'call_year': str(row.get('Call', '')),
                    'practice_areas': str(row.get('Area of Practise + Add Info', '')),
                    'industry_experience': str(row.get('Industry Experience', '')),
                    'previous_companies': str(row.get('Previous Companies/Firms', '')),
                    'notable_experience': str(row.get('Notable Items/Personal Details', ''))
                },
                'location': str(row.get('Location', '')),
                'languages': str(row.get('Languages', ''))
            }
            lawyers_info.append(lawyer)
    
    return lawyers_info

def get_claude_analysis(query, df):
    """Have Claude analyze the entire dataset and recommend lawyers"""
    
    lawyers_info = format_lawyer_info(df)
    
    # Create a more readable format of the lawyers' information
    lawyers_text = "\n\nAVAILABLE LAWYERS:\n"
    for lawyer in lawyers_info:
        lawyers_text += f"\n{lawyer['name']}:\n"
        lawyers_text += f"- Level/Call Year: {lawyer['expertise']['level']} ({lawyer['expertise']['call_year']})\n"
        lawyers_text += f"- Practice Areas: {lawyer['expertise']['practice_areas']}\n"
        lawyers_text += f"- Industry Experience: {lawyer['expertise']['industry_experience']}\n"
        lawyers_text += f"- Availability: {lawyer['availability']['days_per_week']} days/week, {lawyer['availability']['hours_per_month']}/month\n"
        lawyers_text += f"- Preferred Engagement Types: {lawyer['availability']['engagement_types']}\n"
        if lawyer['availability']['notes'] and lawyer['availability']['notes'].lower() not in ['n/a', 'na', 'none', 'no']:
            lawyers_text += f"- Availability Notes: {lawyer['availability']['notes']}\n"
        if lawyer['expertise']['notable_experience'] and lawyer['expertise']['notable_experience'].lower() not in ['n/a', 'na', 'none', 'no']:
            lawyers_text += f"- Notable Experience: {lawyer['expertise']['notable_experience']}\n"

    prompt = f"""You are a knowledgeable legal staffing professional helping match lawyers to client needs. A client has the following need:

{query}

Based on the available lawyers' information below, please:
1. Analyze which lawyers would be the best fit
2. Explain your reasoning
3. Provide between 2-5 clear recommendations, ranked in order of best fit
4. For each recommended lawyer, include their specific availability details

{lawyers_text}

Please consider both expertise match and actual availability. Focus on finding the best practical matches for the client's needs.

Format your response as:

ANALYSIS:
[Your detailed analysis of the best matches]

TOP RECOMMENDATIONS:
1. [First choice with detailed rationale]
2. [Second choice with detailed rationale]
[Include up to 5 total recommendations, each with clear rationale]

AVAILABILITY DETAILS:
[List each recommended lawyer's specific availability, including:
- Days per week available
- Hours per month available
- Any upcoming vacation or availability restrictions
- Preferred engagement types]

ALTERNATIVE CONSIDERATIONS:
[Any other lawyers worth considering and why]"""

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
        st.error(f"Error getting Claude's analysis: {e}")
        return "Error analyzing lawyers. Please try again."

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
    if 'search_triggered' not in st.session_state:
        st.session_state.search_triggered = False

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
        button_key = f"example_{i}"
        if i % 2 == 0:
            if col1.button(f"🔍 {example}", key=button_key):
                st.session_state.query = example
                st.session_state.search_triggered = True
        else:
            if col2.button(f"🔍 {example}", key=button_key):
                st.session_state.query = example
                st.session_state.search_triggered = True

    # Custom query input
    query = st.text_area(
        "Or describe what you're looking for:",
        value=st.session_state.query,
        help="Be as specific as you can about your needs"
    )

    col1, col2 = st.columns([1, 4])
    if col1.button("Find Lawyers"):
        st.session_state.query = query
        st.session_state.search_triggered = True
    
    if col2.button("Clear and Start Over"):
        st.session_state.query = ''
        st.session_state.search_triggered = False
        st.rerun()

    # Perform search when triggered
    if st.session_state.search_triggered and st.session_state.query:
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
            elif section.startswith('AVAILABILITY DETAILS:'):
                with st.expander("Availability Details for Recommended Lawyers", expanded=True):
                    st.write(section.replace('AVAILABILITY DETAILS:', '').strip())
            elif section.startswith('ALTERNATIVE CONSIDERATIONS:'):
                with st.expander("Alternative Options"):
                    st.write(section.replace('ALTERNATIVE CONSIDERATIONS:', '').strip())
    elif not st.session_state.query:
        st.info("Please enter your requirements or select an example query.")

if __name__ == "__main__":
    main()
