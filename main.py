import streamlit as st
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import os

# Load environment variables and setup Anthropic
load_dotenv()
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def get_practice_areas(lawyers_df):
    """Extract unique practice areas from Summary and Expertise"""
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
                **Contact:**  
                {lawyer['Work Email']}
                
                **Education:**  
                {lawyer['Education']}
                
                **Expertise:**  
                • {lawyer['Summary and Expertise'].replace(', ', '\n• ')}
                """)

def get_claude_response(query, lawyers_df):
    """Get Claude's analysis of the best lawyer matches"""
    summary_text = "Available Lawyers and Their Expertise:\n\n"
    for _, lawyer in lawyers_df.iterrows():
        summary_text += f"- {lawyer['Attorney']}\n"
        summary_text += f"  Education: {lawyer['Education']}\n"
        summary_text += f"  Expertise: {lawyer['Summary and Expertise']}\n\n"

    prompt = f"""You are a legal staffing assistant. Your task is to match client needs with available lawyers based on their expertise and background.

Client Need: {query}

{summary_text}

Please analyze the lawyers' profiles and provide the best 3-5 matches in a structured format suitable for creating a table. Format your response exactly like this example, maintaining the exact delimiter structure:

MATCH_START
Rank: 1
Name: John Smith
Key Expertise: Corporate Law, M&A
Education: Harvard Law School J.D.
Recommendation Reason: Extensive experience in corporate transactions with emphasis on technology sector
MATCH_END

Important guidelines:
- Provide 3-5 matches only
- Keep the Recommendation Reason specific but concise (max 150 characters)
- Focus on matching expertise to the client's specific needs
- Use the exact delimiters shown above"""

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
        # Only include columns that exist in the DataFrame
        existing_columns = [col for col in desired_columns if col in df.columns]
        df = df[existing_columns]
        if 'Rank' in df.columns:
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
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No matching lawyers found for your specific needs. Try adjusting your search criteria.")

def main():
    st.title("🧑‍⚖️ Outside GC Lawyer Matcher")
    
    try:
        # Load data with only the columns we know exist
        lawyers_df = pd.read_csv('Cleaned_Matters_OGC.csv')[['Attorney', 'Work Email', 'Education', 'Summary and Expertise']]
        
        # Sidebar filters
        st.sidebar.title("Filters")
        
        # Get practice areas
        practice_areas = []
        for expertise in lawyers_df['Summary and Expertise'].dropna():
            practice_areas.extend([area.strip() for area in expertise.split(',')])
        practice_areas = sorted(list(set(practice_areas)))
        
        # Practice area filter
        selected_practice_area = st.sidebar.selectbox(
            "Practice Area",
            ["All"] + practice_areas
        )
        
        # Main content area
        st.write("### How can we help you find the right lawyer?")
        st.write("Tell us about your legal needs and we'll match you with the best available lawyers.")
        
        # Example queries
        examples = [
            "I need a lawyer experienced in intellectual property and software licensing",
            "Looking for someone who handles business startups and corporate governance",
            "Need help with technology transactions and SaaS agreements",
            "Who would be best for mergers and acquisitions in the technology sector?"
        ]
        
        # Example query buttons
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

        # Filter lawyers based on selection
        filtered_df = lawyers_df.copy()
        if selected_practice_area != "All":
            filtered_df = filtered_df[
                filtered_df['Summary and Expertise'].str.contains(selected_practice_area, na=False, case=False)
            ]
        
        # Custom query input
        query = st.text_area(
            "For more specific matching, describe what you're looking for:",
            value=st.session_state.get('query', ''),
            placeholder="Example: I need help with intellectual property and software licensing...",
            height=100
        )

        # Search and Clear buttons
        col1, col2 = st.columns([1, 4])
        search = col1.button("🔎 Search")
        clear = col2.button("Clear")

        if clear:
            st.session_state.query = ''
            st.rerun()

        # Show counts
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Showing:** {len(filtered_df)} lawyers")
        
        # Show recommendations or all lawyers
        if search and query:
            display_recommendations(query, filtered_df)
        else:
            create_lawyer_cards(filtered_df)
            
    except FileNotFoundError:
        st.error("Could not find the required data file. Please check the file location.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if st.sidebar.checkbox("Show Debug Info"):
            st.sidebar.error(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()
