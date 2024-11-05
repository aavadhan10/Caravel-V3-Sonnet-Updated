import streamlit as st
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import os

# Load environment variables and setup Anthropic
load_dotenv()
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Constants for column names
COLUMNS = ['Attorney', 'Work Email', 'Education', 'Summary and Expertise']

def load_data():
    """Load and validate data"""
    try:
        df = pd.read_csv('Cleaned_Matters_OGC.csv')
        # Print columns for debugging
        st.sidebar.write("DEBUG - Available columns:", list(df.columns))
        return df[COLUMNS]
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def query_claude_with_data(query, lawyers_df):
    """Query Claude with the data we have"""
    lawyer_info = lawyers_df[['Attorney', 'Work Email', 'Education', 'Summary and Expertise']]
    
    context = "Available Lawyers:\n\n"
    for _, row in lawyer_info.iterrows():
        context += f"Lawyer: {row['Attorney']}\n"
        context += f"Email: {row['Work Email']}\n"
        context += f"Education: {row['Education']}\n"
        context += f"Expertise: {row['Summary and Expertise']}\n\n"

    prompt = f"""As a legal staffing assistant, analyze these lawyers' profiles and recommend the best matches for this need:

{query}

{context}

Provide 3-5 recommendations in this exact format:

MATCH_START
Rank: 1
Name: [Attorney Name]
Key Expertise: [Key relevant areas from their expertise]
Education: [Their education]
Recommendation Reason: [Brief explanation of match]
MATCH_END"""

    try:
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return parse_claude_response(response.content[0].text)
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None

def parse_claude_response(response):
    """Parse Claude's response into a structured format"""
    matches = []
    for match in response.split('MATCH_START')[1:]:
        if 'MATCH_END' in match:
            match_text = match.split('MATCH_END')[0].strip()
            match_data = {}
            for line in match_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    match_data[key.strip()] = value.strip()
            if match_data:
                matches.append(match_data)
    
    if matches:
        df = pd.DataFrame(matches)
        if 'Rank' in df.columns:
            df['Rank'] = pd.to_numeric(df['Rank'])
            df = df.sort_values('Rank')
        return df
    return pd.DataFrame()

def display_lawyer_cards(lawyers_df):
    """Display lawyer information in cards"""
    cols = st.columns(3)
    for idx, (_, lawyer) in enumerate(lawyers_df.iterrows()):
        with cols[idx % 3]:
            with st.expander(f"🧑‍⚖️ {lawyer['Attorney']}", expanded=False):
                st.markdown(f"""
                **Contact:**  
                {lawyer['Work Email']}
                
                **Education:**  
                {lawyer['Education']}
                
                **Expertise:**  
                • {lawyer['Summary and Expertise'].replace(', ', '\n• ')}
                """)

def main():
    st.title("🧑‍⚖️ Outside GC Lawyer Matcher")
    
    # Load data
    lawyers_df = load_data()
    if lawyers_df is None:
        st.error("Failed to load lawyer data.")
        return
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Extract practice areas
    all_areas = set()
    for areas in lawyers_df['Summary and Expertise'].dropna():
        all_areas.update(area.strip() for area in areas.split(','))
    practice_areas = sorted(list(all_areas))
    
    # Practice area filter
    selected_area = st.sidebar.selectbox(
        "Filter by Practice Area",
        ["All"] + practice_areas
    )
    
    # Main content
    st.write("### How can we help you find the right lawyer?")
    
    # Example queries
    examples = [
        "I need help with intellectual property and software licensing",
        "Looking for expertise in business startups and corporate governance",
        "Need assistance with technology transactions and SaaS agreements",
        "Who would be best for mergers and acquisitions?"
    ]
    
    # Create example buttons
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(f"🔍 {example}"):
            st.session_state.query = example
            st.rerun()
    
    # Search box
    query = st.text_area(
        "Describe what you're looking for:",
        value=st.session_state.get('query', ''),
        placeholder="Example: I need help with intellectual property matters...",
        height=100
    )
    
    # Filter lawyers
    filtered_df = lawyers_df
    if selected_area != "All":
        filtered_df = lawyers_df[
            lawyers_df['Summary and Expertise'].str.contains(
                selected_area, case=False, na=False
            )
        ]
    
    # Search buttons
    col1, col2 = st.columns([1, 4])
    search = col1.button("🔎 Search")
    clear = col2.button("Clear")
    
    if clear:
        st.session_state.query = ''
        st.rerun()
    
    # Show results
    if search and query:
        matches_df = query_claude_with_data(query, filtered_df)
        if matches_df is not None and not matches_df.empty:
            st.markdown("### 🎯 Top Matches")
            st.dataframe(
                matches_df,
                hide_index=True,
                use_container_width=True
            )
    else:
        display_lawyer_cards(filtered_df)
    
    # Show count in sidebar
    st.sidebar.markdown(f"**Showing:** {len(filtered_df)} lawyers")

if __name__ == "__main__":
    main()
