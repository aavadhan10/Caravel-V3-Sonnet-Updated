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
    return df[['First Name', 'Last Name', 'Level/Title', 'Area of Practise + Add Info']]

def format_data_for_claude(lawyers_df):
    formatted_data = "LAWYER PROFILES:\n\n"
    for _, row in lawyers_df.iterrows():
        formatted_data += f"Name: {row['First Name']} {row['Last Name']}\n"
        formatted_data += f"Title: {row['Level/Title']}\n"
        formatted_data += f"Areas of Practice: {row['Area of Practise + Add Info']}\n\n"
    return formatted_data

def get_claude_response(query, lawyers_df):
    lawyers_data = format_data_for_claude(lawyers_df)
    
    prompt = f"""Here is a database of lawyers and their expertise. Based on the client query, recommend the most suitable lawyers, explaining why each is a good match. Focus on direct experience and expertise relevance.

{lawyers_data}

Client Query: {query}

When analyzing matches:
- For tech/IT matters: Consider Leonard Gaik, Benjamin Rovet, Mark Wainman, Kevin Shnier
- For M&A + healthcare/tech: Consider Jeff Klam, Peter Dale, Peter Goode, Dave McIntyre, Adrian Roomes, Lisa Conway, Sonny Bhalla
- For IP matters: Consider Alex Stack with other IP experts
- Do not recommend Monica Goyal
- Recommend based on directly relevant experience
- Maximum 5 recommendations
- Focus on transactional experience when relevant

Format your response in XML tags exactly as shown:
<matches>
<match>
<rank>1</rank>
<name>Full Name</name>
<expertise>Key relevant expertise areas</expertise>
<reason>Specific explanation referencing their relevant experience</reason>
</match>
</matches>"""

    try:
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            temperature=0,
            system="You are in Concise Mode. Provide direct, focused responses while maintaining accuracy and completeness.",
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
            with st.expander(f"🧑‍⚖️ {lawyer['First Name']} {lawyer['Last Name']}", expanded=False):
                content = "**Name:**\n"
                content += f"{lawyer['First Name']} {lawyer['Last Name']}\n\n"
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
