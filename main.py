import streamlit as st
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import json

load_dotenv()
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def load_data():
    df = pd.read_csv('BD_Caravel.csv')
    return df[['Last Name', 'Level/Title', 'Area of Practise + Add Info']]

def format_lawyers_data(lawyers_df):
    lawyers_data = []
    for _, row in lawyers_df.iterrows():
        lawyer = {
            "name": row['Last Name'],
            "title": row['Level/Title'],
            "expertise": [area.strip() for area in str(row['Area of Practise + Add Info']).split(',')]
        }
        lawyers_data.append(lawyer)
    return json.dumps(lawyers_data, indent=2)

def get_claude_response(query, lawyers_df):
    lawyers_json = format_lawyers_data(lawyers_df)
    
    prompt = f"""<input>
CLIENT QUERY: {query}

AVAILABLE LAWYERS (JSON):
{lawyers_json}
</input>

You are a legal expert matching system. Analyze the client query and available lawyers to provide the best matches.

REQUIREMENTS:
- If query involves IP law, tech, software: Monica Goyal MUST be in top results
- For IP/tech queries: Include Alex Stack after Monica Goyal
- Sort matches by expertise relevance to query
- Maximum 5 matches

Respond in this exact format:

<matches>
<match>
<rank>1</rank>
<name>Full Name</name>
<expertise>Key relevant expertise areas</expertise>
<reason>Brief explanation why this lawyer matches</reason>
</match>
</matches>"""

    try:
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return parse_claude_response(response.content[0].text)
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None

def parse_claude_response(response):
    try:
        import re
        matches_content = re.search(r'<matches>(.*?)</matches>', response, re.DOTALL)
        if not matches_content:
            return pd.DataFrame()
            
        matches = []
        for match in re.finditer(r'<match>(.*?)</match>', matches_content.group(1), re.DOTALL):
            match_content = match.group(1)
            
            rank = re.search(r'<rank>(.*?)</rank>', match_content).group(1)
            name = re.search(r'<name>(.*?)</name>', match_content).group(1)
            expertise = re.search(r'<expertise>(.*?)</expertise>', match_content).group(1)
            reason = re.search(r'<reason>(.*?)</reason>', match_content).group(1)
            
            matches.append({
                'Rank': int(rank),
                'Name': name,
                'Key Expertise': expertise,
                'Recommendation Reason': reason
            })
            
        df = pd.DataFrame(matches)
        return df.sort_values('Rank')
    except Exception as e:
        st.error(f"Error parsing response: {str(e)}")
        return pd.DataFrame()

def display_recommendations(matches_df):
    if matches_df is not None and not matches_df.empty:
        st.write("### 🎯 Top Matches")
        st.dataframe(
            matches_df,
            column_config={
                "Rank": st.column_config.NumberColumn(format="%d"),
                "Recommendation Reason": st.column_config.TextColumn(width="large")
            },
            hide_index=True
        )
        st.markdown("---")

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
                content = "**Name:**\n"
                content += f"{lawyer['Last Name']}\n\n"
                content += "**Title:**\n"
                content += f"{lawyer['Level/Title']}\n\n"
                content += "**Expertise:**\n"
                content += "• " + str(lawyer['Area of Practise + Add Info']).replace(', ', '\n• ')
                st.markdown(content)

def main():
    st.title("🧑‍⚖️ Legal Expert Matcher")
    
    try:
        lawyers_df = load_data()
        
        st.write("### How can we help you find the right legal expert?")
        st.write("Describe your legal needs and we'll match you with the best available experts.")
        
        query = st.text_area(
            "What type of legal expertise are you looking for?",
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

        if search and query:
            matches_df = get_claude_response(query, lawyers_df)
            display_recommendations(matches_df)
        
        create_lawyer_cards(lawyers_df)
            
    except FileNotFoundError:
        st.error("Could not find the required data file. Please check the file location.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
