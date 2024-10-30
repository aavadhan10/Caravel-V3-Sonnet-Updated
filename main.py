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
    try:
        availability_df = pd.read_csv('Caravel Law Availability - October 18th, 2024.csv')
        bios_df = pd.read_csv('BD_Caravel.csv')
        
        # Clean up and merge data
        availability_df['Name'] = availability_df['What is your name?'].str.strip()
        bios_df['Name'] = bios_df['First Name'].str.strip() + ' ' + bios_df['Last Name'].str.strip()
        
        merged_df = pd.merge(availability_df, bios_df, on='Name', how='inner')
        st.sidebar.write(f"Debug: Successfully loaded {len(merged_df)} lawyers")
        return merged_df
    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        return None

def prepare_lawyer_data(df):
    """Prepare lawyer information for analysis"""
    lawyer_info = []
    
    for _, row in df.iterrows():
        # Basic lawyer information
        info = {
            'name': row['Name'],
            'level': row.get('Level/Title', ''),
            'call_year': row.get('Call', ''),
            'practice_areas': row.get('Area of Practise + Add Info', ''),
            'industry_experience': row.get('Industry Experience', ''),
            'location': row.get('Location', ''),
            'availability': {
                'status': row.get('Do you have capacity to take on new work?', ''),
                'days_per_week': row.get('What is your capacity to take on new work for the forseeable future? Days per week', ''),
                'hours_per_month': row.get('What is your capacity to take on new work for the foreseeable future? Hours per month', ''),
                'notes': row.get('Do you have any comments or instructions you should let us know about that may impact your short/long-term availability? For instance, are you going on vacation (please provide exact dates)?', ''),
                'engagement_types': row.get('What type of engagement would you like to consider?', '')
            },
            'experience': {
                'previous_companies': row.get('Previous Companies/Firms', ''),
                'notable_items': row.get('Notable Items/Personal Details', '')
            }
        }
        
        # Only include if they have some availability
        if str(info['availability']['status']).lower() in ['yes', 'maybe']:
            lawyer_info.append(info)
    
    st.sidebar.write(f"Debug: Found {len(lawyer_info)} available lawyers")
    return lawyer_info

def get_claude_analysis(query, lawyer_data):
    """Get Claude's analysis of the best lawyer matches"""
    
    # Format lawyer information for prompt
    lawyer_descriptions = []
    for lawyer in lawyer_data:
        desc = f"\nLAWYER: {lawyer['name']}\n"
        desc += f"Level/Call Year: {lawyer['level']} ({lawyer['call_year']})\n"
        desc += f"Practice Areas: {lawyer['practice_areas']}\n"
        desc += f"Industry Experience: {lawyer['industry_experience']}\n"
        desc += f"Location: {lawyer['location']}\n"
        desc += "Availability:\n"
        desc += f"- Days per week: {lawyer['availability']['days_per_week']}\n"
        desc += f"- Hours per month: {lawyer['availability']['hours_per_month']}\n"
        desc += f"- Engagement types: {lawyer['availability']['engagement_types']}\n"
        if lawyer['availability']['notes']:
            desc += f"- Notes: {lawyer['availability']['notes']}\n"
        if lawyer['experience']['notable_items']:
            desc += f"Notable Experience: {lawyer['experience']['notable_items']}\n"
        lawyer_descriptions.append(desc)

    prompt = f"""You are assisting with finding the best lawyer match for a client's needs. Please analyze this request carefully and recommend the most suitable lawyers based on their expertise and availability.

CLIENT NEED:
{query}

AVAILABLE LAWYERS:
{''.join(lawyer_descriptions)}

Please analyze this request as a legal staffing professional would, considering:
1. Direct expertise match with the client's needs
2. Relevant industry experience
3. Availability and engagement type alignment
4. Years of experience and seniority
5. Location and practical considerations

Provide your response in this format:

ANALYSIS:
[A thoughtful analysis of why certain lawyers would be good matches, considering both expertise and practical factors]

RECOMMENDED LAWYERS:
[2-5 ranked recommendations, each with:
- Name and key qualifications
- Specific reasons they're a good match
- Any important considerations about their availability]

AVAILABILITY DETAILS:
[Specific availability information for each recommended lawyer]

ADDITIONAL SUGGESTIONS:
[Any other lawyers worth considering and why]"""

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
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        st.error("Please set the ANTHROPIC_API_KEY environment variable")
        return
    
    # Load data
    df = load_data()
    if df is None:
        return
        
    # Initialize session state
    if 'query' not in st.session_state:
        st.session_state.query = ''
    if 'search_triggered' not in st.session_state:
        st.session_state.search_triggered = False

    # Example queries
    st.write("### How can we help you find the right lawyer?")
    examples = [
        "I need a lawyer with trademark and IP experience who can start work soon",
        "Looking for someone experienced in employment law to help with HR policies and contracts",
        "Need a lawyer experienced with startups, financing, and corporate structure",
        "Who would be best for drafting and negotiating SaaS agreements and technology contracts?"
    ]
    
    # Example query buttons in two columns
    col1, col2 = st.columns(2)
    for i, example in enumerate(examples):
        if i % 2 == 0:
            if col1.button(f"🔍 {example}", key=f"example_{i}"):
                st.session_state.query = example
                st.session_state.search_triggered = True
        else:
            if col2.button(f"🔍 {example}", key=f"example_{i}"):
                st.session_state.query = example
                st.session_state.search_triggered = True

    # Custom query input
    st.write("\n### Or describe your specific needs:")
    query = st.text_area(
        "",
        value=st.session_state.query,
        help="Be specific about your legal needs, industry context, and any timing requirements",
        placeholder="E.g., I need a lawyer experienced in technology licensing and privacy law for a SaaS company..."
    )

    # Search and Clear buttons
    col1, col2 = st.columns([1, 4])
    search = col1.button("Find Lawyers")
    clear = col2.button("Clear and Start Over")

    if clear:
        st.session_state.query = ''
        st.session_state.search_triggered = False
        st.rerun()

    if search:
        st.session_state.query = query
        st.session_state.search_triggered = True

    # Process search
    if st.session_state.search_triggered and st.session_state.query:
        lawyer_data = prepare_lawyer_data(df)
        
        if not lawyer_data:
            st.error("No available lawyers found in the database.")
            return
            
        with st.spinner("Analyzing available lawyers..."):
            analysis = get_claude_analysis(st.session_state.query, lawyer_data)
        
        if analysis:
            # Display results in a clean format
            sections = analysis.split('\n\n')
            for section in sections:
                if section.startswith('ANALYSIS:'):
                    with st.expander("Detailed Analysis", expanded=True):
                        st.write(section.replace('ANALYSIS:', '').strip())
                elif section.startswith('RECOMMENDED LAWYERS:'):
                    st.write("### Recommended Lawyers")
                    st.write(section.replace('RECOMMENDED LAWYERS:', '').strip())
                elif section.startswith('AVAILABILITY DETAILS:'):
                    with st.expander("Availability Information", expanded=True):
                        st.write(section.replace('AVAILABILITY DETAILS:', '').strip())
                elif section.startswith('ADDITIONAL SUGGESTIONS:'):
                    with st.expander("Additional Options"):
                        st.write(section.replace('ADDITIONAL SUGGESTIONS:', '').strip())
    elif not st.session_state.query:
        st.info("Please enter your requirements or select an example query.")

if __name__ == "__main__":
    main()
