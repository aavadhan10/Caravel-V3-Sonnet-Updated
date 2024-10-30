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

def get_claude_analysis(query, df):
    """Have Claude analyze the entire dataset and recommend lawyers"""
    
    # Convert relevant dataframe information to a structured format
    lawyers_info = []
    for _, row in df.iterrows():
        lawyer_info = {
            'name': row['Name'],
            'availability': {
                'has_capacity': row['Do you have capacity to take on new work?'],
                'days_per_week': row['What is your capacity to take on new work for the forseeable future? Days per week'],
                'hours_per_month': row['What is your capacity to take on new work for the foreseeable future? Hours per month'],
                'notes': row['Do you have any comments or instructions you should let us know about that may impact your short/long-term availability? For instance, are you going on vacation (please provide exact dates)?']
            },
            'expertise': {
                'practice_areas': row['Area of Practise + Add Info'],
                'industry_experience': row['Industry Experience'],
                'call_year': row['Call'],
                'notable_experience': row['Notable Items/Personal Details']
            }
        }
        lawyers_info.append(lawyer_info)

    prompt = f"""You are a highly knowledgeable legal staffing professional with access to a database of lawyers. A client has the following need:

{query}

I'll show you information about available lawyers, and I'd like you to:
1. Analyze which lawyers would be the best fit
2. Explain your reasoning
3. Suggest a clear recommendation
4. Mention any potential alternatives worth considering

Here is the lawyer information:
{lawyers_info}

Please approach this analysis as you would a natural conversation, focusing on the most relevant details and providing clear, practical recommendations. Consider both expertise match and actual availability.

Format your response as:

ANALYSIS:
[Your detailed analysis of the best matches]

TOP RECOMMENDATIONS:
1. [Primary recommendation with clear rationale]
2. [Secondary recommendation if applicable]
3. [Third option if applicable]

ALTERNATIVE CONSIDERATIONS:
[Any other lawyers worth considering and why]

AVAILABILITY NOTES:
[Any important notes about the recommended lawyers' availability]"""

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

    # Example queries
    st.write("### How can we help you find the right lawyer?")
    st.write("Try asking something like:")
    examples = [
        "I need a lawyer with trademark experience who can start work soon",
        "Looking for someone with employment law experience to help with HR policies and contracts",
        "Need a lawyer experienced with startups and financing",
        "Who would be best for drafting and negotiating SaaS agreements?"
    ]
    for example in examples:
        if st.button(f"🔍 {example}"):
            st.session_state.query = example

    # Custom query input
    query = st.text_area(
        "Describe what you're looking for:",
        value=st.session_state.get('query', ''),
        help="Be as specific as you can about your needs"
    )

    if st.button("Find Lawyers") or 'query' in st.session_state:
        if query:
            with st.spinner("Analyzing available lawyers..."):
                analysis = get_claude_analysis(query, df)
            
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
        else:
            st.warning("Please enter your requirements or select an example query.")

    # Clear the session state if needed
    if st.button("Clear and Start Over"):
        st.session_state.pop('query', None)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
