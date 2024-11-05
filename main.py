import streamlit as st
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def clean_text_field(text):
    """Clean and standardize text fields"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return text.strip()

def get_practice_areas(lawyers_df):
    """Extract unique practice areas from lawyers summary"""
    all_areas = set()
    for areas in lawyers_df['Summary and Expertise'].dropna():
        areas_list = [area.strip() for area in areas.split(',')]
        all_areas.update(areas_list)
    return sorted(list(all_areas))

def create_lawyer_cards(lawyers_df):
    """Create card layout for lawyers"""
    if lawyers_df.empty:
        st.warning("No lawyers match the selected filters.")
        return
        
    st.write("### 📊 Available Lawyers")
    
    lawyers_df = lawyers_df.sort_values('Attorney')
    cols = st.columns(3)
    
    for idx, (_, lawyer) in enumerate(lawyers_df.iterrows()):
        with cols[idx % 3]:
            with st.expander(f"🧑‍⚖️ {lawyer['Attorney']}", expanded=False):
                st.markdown(f"""
                **Email:**  
                {lawyer['Work Email']}
                
                **Education:**  
                {lawyer['Education']}
                
                **Areas of Expertise:**  
                {' • '.join(lawyer['Summary and Expertise'].split(', '))}
                """)

def get_claude_response(query, lawyers_df):
    """Get Claude's analysis of the best lawyer matches"""
    summary_text = "Available Lawyers and Their Expertise:\n\n"
    for _, lawyer in lawyers_df.iterrows():
        summary_text += f"- {lawyer['Attorney']}\n"
        summary_text += f"  Education: {lawyer['Education']}\n"
        summary_text += f"  Expertise: {lawyer['Summary and Expertise']}\n\n"

    prompt = f"""You are a legal staffing assistant. Your task is to match client needs with available lawyers based on their expertise and education.

Client Need: {query}

{summary_text}

Please analyze the lawyers' profiles and provide the best 3-5 matches in a structured format suitable for creating a table. Format your response exactly like this example, maintaining the exact delimiter structure:

MATCH_START
Rank: 1
Name: John Smith
Key Expertise: Corporate Law, M&A
Education: Harvard Law School J.D.
Recommendation Reason: 15+ years handling similar corporate transactions with emphasis on tech sector
MATCH_END

MATCH_START
Rank: 2
[Continue with next match]

Important guidelines:
- Provide 3-5 matches only
- Keep the Recommendation Reason specific but concise (max 150 characters)
- Focus on concrete experience and specific expertise that matches the client's needs
- Use the exact delimiters shown above
- Only include lawyers whose expertise truly matches the needs"""

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
    """Parse Claude's response into a structured format for table display"""
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
        desired_columns = ['Rank', 'Name', 'Key Expertise', 'Education', 'Recommendation Reason']
        df = df[desired_columns]
        df['Rank'] = pd.to_numeric(df['Rank'])
        df = df.sort_values('Rank')
    
    return df

def display_recommendations(query, lawyers_df):
    """Display lawyer recommendations in a formatted table"""
    with st.spinner("Finding the best matches..."):
        results_df = get_claude_response(query, lawyers_df)
        
        if results_df is not None and not results_df.empty:
            st.markdown("### 🎯 Top Lawyer Matches")
            
            st.dataframe(
                results_df,
                column_config={
                    "Rank": st.column_config.NumberColumn(
                        "Rank",
                        help="Match ranking",
                        format="%d",
                        width="small"
                    ),
                    "Name": st.column_config.Column(
                        "Name",
                        help="Lawyer name",
                        width="medium"
                    ),
                    "Key Expertise": st.column_config.Column(
                        "Key Expertise",
                        help="Relevant areas of expertise",
                        width="large"
                    ),
                    "Education": st.column_config.Column(
                        "Education",
                        help="Educational background",
                        width="medium"
                    ),
                    "Recommendation Reason": st.column_config.Column(
                        "Recommendation Reason",
                        help="Why this lawyer is specifically recommended for your needs",
                        width="large"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No matching lawyers found for your specific needs. Try adjusting your search criteria.")

def main():
    st.title("🧑‍⚖️ Legal Counsel Matcher")
    
    try:
        # Load the data file
        lawyers_df = pd.read_csv('Cleaned_Matters_OGC.csv')
        
        # Move filters to sidebar
        st.sidebar.title("Filters")
        
        # Debug checkbox in sidebar
        show_debug = st.sidebar.checkbox("Show Debug Information", False)
        
        if show_debug:
            st.sidebar.write("### Data Preview")
            st.sidebar.write(lawyers_df.head())
        
        # Get all unique practice areas
        all_practice_areas = get_practice_areas(lawyers_df)
        selected_practice_area = st.sidebar.selectbox(
            "Practice Area",
            ["All"] + all_practice_areas
        )
        
        # Main content area
        st.write("### How can we help you find the right lawyer?")
        st.write("Tell us about your legal needs and we'll match you with the best available lawyers.")
        
        # Example queries
        examples = [
            "I need a lawyer with intellectual property experience for software licensing",
            "Looking for someone experienced in business startups and corporate governance",
            "Need a lawyer experienced with data privacy and cybersecurity",
            "Who would be best for technology transactions and SaaS agreements?"
        ]
        
        # Example query buttons in two columns
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

        st.write("")
        
        # Filter lawyers based on selection
        filtered_df = lawyers_df.copy()
        
        if selected_practice_area != "All":
            filtered_df = filtered_df[
                filtered_df['Summary and Expertise'].str.contains(selected_practice_area, na=False)
            ]
        
        # Custom query input
        query = st.text_area(
            "For more specific matching, describe what you're looking for (optional):",
            value=st.session_state.get('query', ''),
            placeholder="Example: I need help with intellectual property and software licensing...",
            height=100
        )

        # Search and Clear buttons
        button_col1, button_col2 = st.columns([1, 4])
        search = button_col1.button("🔎 Search")
        clear = button_col2.button("Clear")

        if clear:
            st.session_state.query = ''
            st.rerun()

        # Show filtered results counts
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Current Filters")
        st.sidebar.markdown(f"**Practice Area:** {selected_practice_area}")
        st.sidebar.markdown(f"**Total Lawyers:** {len(filtered_df)}")

        # Show Claude's recommendations when search is used
        if search and query:
            st.session_state.query = query
            display_recommendations(query, filtered_df)
        
        # Show all lawyers if no filters are applied and no search is performed
        if not (search and query):
            create_lawyer_cards(filtered_df)
            
    except FileNotFoundError as e:
        st.error("Could not find the required data file. Please check your data file location.")
        if show_debug:
            st.sidebar.write("Error details:", str(e))
        return
    except Exception as e:
        st.error("An error occurred while processing the data.")
        if show_debug:
            st.sidebar.write("Error details:", str(e))
        return

if __name__ == "__main__":
    main()
    
