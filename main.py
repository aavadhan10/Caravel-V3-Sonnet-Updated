import streamlit as st
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import json

load_dotenv()
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def load_data():
    # Load main lawyer data
    df = pd.read_csv('BD_Caravel.csv')
    
    # Load availability data
    try:
        availability_df = pd.read_csv('Corrected_Caravel_Law_Availability.csv')
        # Convert hours to numeric, handling the '80+' case
        availability_df['Hours'] = availability_df['What is your capacity to take on new work for the foreseeable future? Hours per month'].apply(
            lambda x: 80 if isinstance(x, str) and '+' in x else float(x.split()[0]) if isinstance(x, str) else float(x)
        )
        
        def hours_to_availability(hours):
            if pd.isna(hours):
                return 'High'  # Default value
            if hours >= 80:
                return 'High'
            elif hours >= 40:
                return 'Medium'
            else:
                return 'Low'
        
        availability_df['Availability'] = availability_df['Hours'].apply(hours_to_availability)
        
        # Create full name in both dataframes for merging
        df['Full_Name'] = df['First Name'] + ' ' + df['Last Name']
        availability_df['Full_Name'] = availability_df['What is your name?']
        
        # Merge on full name
        df = pd.merge(
            df,
            availability_df[['Full_Name', 'Availability']],
            on='Full_Name',
            how='left'
        )
        
        # Clean up
        df['Availability'] = df['Availability'].fillna('High')
        df = df.drop(['Full_Name'], axis=1)
        
    except Exception as e:
        print(f"Error loading availability data: {e}")
        df['Availability'] = 'High'
        
    return df[['First Name', 'Last Name', 'Level/Title', 'Area of Practise + Add Info', 'Availability']]

def format_data_for_claude(lawyers_df, availability_filter=None):
    if availability_filter:
        lawyers_df = lawyers_df[lawyers_df['Availability'] == availability_filter]
    
    formatted_data = "LAWYER PROFILES WITH DETAILED EXPERIENCE:\n\n"
    for _, row in lawyers_df.iterrows():
        formatted_data += f"Name: {row['First Name']} {row['Last Name']}\n"
        formatted_data += f"Title: {row['Level/Title']}\n"
        formatted_data += f"Areas of Practice: {row['Area of Practise + Add Info']}\n\n"
    return formatted_data

def get_claude_response(query, lawyers_df, availability_filter=None):
    lawyers_data = format_data_for_claude(lawyers_df, availability_filter)
    
    prompt = f"""Here is a database of lawyers and their expertise. Based on the client query, recommend the most suitable lawyers, explaining why each is a good match.

{lawyers_data}

Client Query: {query}

When analyzing matches, use these specific guidelines:
1. For technology transaction queries:
   - Priority: Leonard Gaik, Kevin Michael Shnier, Benjamin Rovet, Mark Wainman, Jeff Klam, Peter Dale, Peter Goode, Dave McIntyre
   - Focus on their specific tech industry experience

2. For M&A + healthcare/technology:
   - Priority: Adrian Roomes, Ajay Krishnan, Lisa Conway, Sonny Bhalla, Jeff Klam, Peter Dale, Peter Goode, Dave McIntyre
   - Consider their healthcare sector and M&A expertise

3. For technology M&A/acquisition queries:
   - Priority: Leonard Gaik, Kevin Michael Shnier, Peter Torn, Neil Kothari, Jeff Klam, Peter Dale, Peter Goode, Dave McIntyre
   - Focus on their M&A and technology experience

4. For IP matters:
   - Include Alex Stack with other IP experts

Additional rules:
- Do not recommend Monica Goyal
- For any M&A or tech-related queries, always consider Jeff Klam, Peter Dale, Peter Goode, and Dave McIntyre as potential matches
- Maximum 5 recommendations
- Order by relevance to the specific query

Format your response exactly as:
<matches>
<match>
<rank>1</rank>
<name>Full Name</name>
<expertise>Key relevant expertise areas</expertise>
<reason>Specific explanation referencing their exact experience</reason>
</match>
</matches>"""

    try:
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            temperature=0,
            system="Precisely match the lawyers and their experience as shown in the profiles. Always consider Jeff Klam, Peter Dale, Peter Goode, and Dave McIntyre for M&A and tech queries.",
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

def create_lawyer_cards(lawyers_df, availability_filter=None):
    if availability_filter:
        lawyers_df = lawyers_df[lawyers_df['Availability'] == availability_filter]
        
    if lawyers_df.empty:
        st.warning("No lawyers match the selected filters.")
        return
        
    st.write("### 📊 Available Lawyers")
    
    lawyers_df = lawyers_df.sort_values('Last Name')
    cols = st.columns(3)
    
    for idx, (_, lawyer) in enumerate(lawyers_df.iterrows()):
        availability_color = {
            'High': '🟢',
            'Medium': '🟡',
            'Low': '🔴'
        }.get(lawyer['Availability'], '⚪')
        
        with cols[idx % 3]:
            with st.expander(f"🧑‍⚖️ {lawyer['First Name']} {lawyer['Last Name']} {availability_color}", expanded=False):
                content = "**Name:**\n"
                content += f"{lawyer['First Name']} {lawyer['Last Name']}\n\n"
                content += "**Title:**\n"
                content += f"{lawyer['Level/Title']}\n\n"
                content += "**Expertise:**\n"
                content += "• " + str(lawyer['Area of Practise + Add Info']).replace(', ', '\n• ')
                content += f"\n\n**Availability:** {availability_color} {lawyer['Availability']}"
                st.markdown(content)

def main():
    st.title("🧑‍⚖️ Legal Expert Matcher")
    
    try:
        lawyers_df = load_data()
        
        with st.sidebar:
            st.header("Filters")
            availability_options = ['All', 'High', 'Medium', 'Low']
            availability_filter = st.selectbox(
                "Lawyer Availability",
                options=availability_options,
                index=0
            )
            
            st.markdown("""
            **Availability Legend:**
            - 🟢 High (80+ hours/month)
            - 🟡 Medium (40-79 hours/month)
            - 🔴 Low (<40 hours/month)
            """)
        
        selected_availability = None if availability_filter == 'All' else availability_filter
        
        st.write("### How can we help you find the right legal expert?")
        
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
            matches_df = get_claude_response(query, lawyers_df)  # Availability filter doesn't affect search
            display_recommendations(matches_df)
        
        create_lawyer_cards(lawyers_df, selected_availability)
            
    except FileNotFoundError:
        st.error("Could not find the required data file. Please check the file location.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
