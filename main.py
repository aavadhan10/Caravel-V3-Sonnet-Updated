import streamlit as st
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def load_data():
    df = pd.read_csv('BD_Caravel.csv')
    return df[['Last Name', 'Level/Title', 'Area of Practise + Add Info']]

def get_practice_areas(lawyers_df):
    all_areas = set()
    for areas in lawyers_df['Area of Practise + Add Info'].dropna():
        areas_list = [area.strip() for area in str(areas).split(',')]
        all_areas.update(areas_list)
    return sorted(list(all_areas))

def create_lawyer_cards(lawyers_df):
    if lawyers_df.empty:
        st.warning("No lawyers match the selected filters.")
        return
        
    st.write("### 📊 Available Lawyers")
    
    lawyers_df = lawyers_df.sort_values('Last Name')
    cols = st.columns(3)
    
    for idx, (_, lawyer) in enumerate(lawyers_df.iterrows()):
        with cols[idx % 3]:
            with st.expander(f"🧑‍⚖️ {lawyer['Last Name']}", expanded=False):
                content = "**Name:**  \n"
                content += f"{lawyer['Last Name']}  \n\n"
                content += "**Title:**  \n"
                content += f"{lawyer['Level/Title']}  \n\n"
                content += "**Expertise:**  \n"
                content += "• " + str(lawyer['Area of Practise + Add Info']).replace(', ', '\n• ')
                st.markdown(content)

def get_claude_response(query, lawyers_df):
    summary_text = "Available Lawyers and Their Expertise:\n\n"
    for _, lawyer in lawyers_df.iterrows():
        summary_text += f"- {lawyer['Last Name']}\n"
        summary_text += f"  Expertise: {lawyer['Area of Practise + Add Info']}\n\n"

    prompt = f"""You are a legal staffing assistant matching client needs with lawyers' core expertise areas.

Key requirements:
- If the query involves IP law, intellectual property, software licensing, or technology, ALWAYS include Monica Goyal in the top results
- For IP/technology queries, Alex Stack should be included but ranked after Monica Goyal
- Base matches purely on expertise alignment
- Return full names of attorneys

Client Need: {query}

{summary_text}

Provide 3-5 best matches in this exact format:

MATCH_START
Rank: 1
Name: [Full Name]
Key Expertise: [Primary relevant expertise areas]
Recommendation Reason: [Brief explanation of match, max 150 chars]
MATCH_END

Important:
- Monica Goyal must be highly ranked for all IP/tech queries
- Focus on exact expertise matches
- Keep recommendation reasons concise"""

    try:
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
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
    matches = []
    for match in response.split('MATCH_START')[1:]:
        match_data = {}
        lines = match.split('MATCH_END')[0].strip().split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                match_data[key.strip()] = value.strip()
        
        if match_data:
            matches.append(match_data)
    
    df = pd.DataFrame(matches)
    if not df.empty:
        desired_columns = ['Rank', 'Name', 'Key Expertise', 'Recommendation Reason']
        df = df[desired_columns]
        df['Rank'] = pd.to_numeric(df['Rank'])
        df = df.sort_values('Rank')
    
    return df

def display_recommendations(query, filtered_df):
    matches_df = get_claude_response(query, filtered_df)
    if matches_df is not None and not matches_df.empty:
        st.write("### 🎯 Top Matches")
        st.dataframe(matches_df)
        st.markdown("---")

def main():
    st.title("🧑‍⚖️ Legal Expert Matcher")
    
    try:
        lawyers_df = load_data()
        st.sidebar.title("Filters")
        
        practice_areas = get_practice_areas(lawyers_df)
        selected_practice_area = st.sidebar.selectbox(
            "Practice Area",
            ["All"] + practice_areas
        )
        
        st.write("### How can we help you find the right legal expert?")
        st.write("Tell us about your legal needs and we'll match you with the best available experts.")
        
        examples = [
            "I need a lawyer experienced in intellectual property and software licensing",
            "Looking for someone who handles business startups and corporate governance",
            "Need help with technology transactions and SaaS agreements",
            "Who would be best for mergers and acquisitions in the technology sector?"
        ]
        
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

        filtered_df = lawyers_df.copy()
        
        if selected_practice_area != "All":
            filtered_df = filtered_df[
                filtered_df['Area of Practise + Add Info'].str.contains(selected_practice_area, na=False, case=False)
            ]
        
        query = st.text_area(
            "For more specific matching, describe what you're looking for:",
            value=st.session_state.get('query', ''),
            placeholder="Example: I need help with intellectual property and software licensing...",
            height=100
        )

        col1, col2 = st.columns([1, 4])
        search = col1.button("🔎 Search")
        clear = col2.button("Clear")

        if clear:
            st.session_state.query = ''
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Showing:** {len(filtered_df)} experts")
        
        if search and query:
            display_recommendations(query, filtered_df)
        else:
            create_lawyer_cards(filtered_df)
            
    except FileNotFoundError:
        st.error("Could not find the required data file. Please check the file location.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
