import streamlit as st
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def standardize_name(name):
    """Standardize name format for matching"""
    if pd.isna(name) or not isinstance(name, str):
        return ""
    
    name = str(name).lower().strip()
    name = ' '.join([part for part in name.split() if '(' not in part and ')' not in part])
    name_parts = name.split()
    if len(name_parts) > 2:
        name = f"{name_parts[0]} {name_parts[-1]}"
    name = name.replace('- ', '-').replace(' -', '-')
    last_name = name.split()[-1]
    if last_name.endswith('s'):
        name = f"{' '.join(name.split()[:-1])} {last_name[:-1]}"
    return name

def clean_text_field(text):
    """Clean and standardize text fields"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return text.strip()

def get_practice_areas(lawyers_summary):
    """Extract unique practice areas from lawyers summary"""
    all_areas = set()
    for lawyer in lawyers_summary:
        areas = [area.strip() for area in lawyer['practice_areas'].split(',') if area.strip()]
        all_areas.update(areas)
    return sorted(list(all_areas))

def create_lawyer_cards(lawyers_summary):
    """Create card layout for lawyers"""
    if not lawyers_summary:
        st.warning("No lawyers match the selected filters.")
        return
    
    lawyers_summary = sorted(lawyers_summary, key=lambda x: x['name'])
    cols = st.columns(3)
    
    for idx, lawyer in enumerate(lawyers_summary):
        with cols[idx % 3]:
            with st.expander(f"🧑‍⚖️ {lawyer['name']}", expanded=False):
                st.markdown(f"""
                **Availability Status:**  
                {lawyer['availability_status']}
                
                **Schedule:**
                - {lawyer['days_available']} days/week
                - {lawyer['hours_available']}/month
                
                **Practice Areas:**  
                {format_practice_areas(lawyer['practice_areas']).replace('      •', '•')}
                
                **Industry Experience:**  
                {format_practice_areas(lawyer['experience']).replace('      •', '•')}
                """)

def prepare_lawyer_summary(availability_data, bios_data, show_debug=False):
    """Create a concise summary of lawyer information"""
    if show_debug:
        st.sidebar.write("### Debug Information")
        st.sidebar.write("Initial data shapes:")
        st.sidebar.write(f"Availability data: {availability_data.shape}")
        st.sidebar.write(f"Bios data: {bios_data.shape}")
    
    availability_data['Original_Name'] = availability_data['What is your name?'].copy()
    availability_data['Standardized_Name'] = availability_data['What is your name?'].apply(standardize_name)
    
    bios_data['Original_Name'] = bios_data['First Name'] + ' ' + bios_data['Last Name']
    bios_data['Standardized_Name'] = bios_data.apply(
        lambda x: standardize_name(f"{x['First Name']} {x['Last Name']}"), axis=1)
    
    if show_debug:
        st.sidebar.write("\n### Name Standardization Results")
        st.sidebar.write("\nAvailability Data Names:")
        name_comparison = pd.DataFrame({
            'Original': availability_data['Original_Name'],
            'Standardized': availability_data['Standardized_Name']
        })
        st.sidebar.write(name_comparison)
        
        st.sidebar.write("\nBios Data Names:")
        name_comparison_bios = pd.DataFrame({
            'Original': bios_data['Original_Name'],
            'Standardized': bios_data['Standardized_Name']
        })
        st.sidebar.write(name_comparison_bios)
    
    lawyers_summary = []
    capacity_column = 'Do you have capacity to take on new work?'
    
    available_lawyers = availability_data[
        availability_data[capacity_column].fillna('').str.lower().str.contains('yes|maybe|y|m', na=False)
    ]
    
    if show_debug:
        st.sidebar.write(f"\nNumber of available lawyers found: {len(available_lawyers)}")
    
    for _, row in available_lawyers.iterrows():
        std_name = row['Standardized_Name']
        bio_row = bios_data[bios_data['Standardized_Name'] == std_name]
        
        if not bio_row.empty:
            if show_debug:
                st.sidebar.write(f"Match found: {row['Original_Name']} ↔ {bio_row.iloc[0]['Original_Name']}")
            bio_row = bio_row.iloc[0]
            
            summary = {
                'name': row['Original_Name'],
                'availability_status': clean_text_field(row[capacity_column]),
                'practice_areas': clean_text_field(bio_row.get('Area of Practise + Add Info', '')),
                'experience': clean_text_field(bio_row.get('Industry Experience', '')),
                'days_available': clean_text_field(row.get(
                    'What is your capacity to take on new work for the forseeable future? Days per week', 
                    'Not specified'
                )),
                'hours_available': clean_text_field(row.get(
                    'What is your capacity to take on new work for the foreseeable future? Hours per month',
                    'Not specified'
                ))
            }
            lawyers_summary.append(summary)
    
    if show_debug:
        st.sidebar.write(f"\nFinal number of matched lawyers: {len(lawyers_summary)}")
    return lawyers_summary

def format_practice_areas(practice_areas):
    """Format practice areas into a bulleted list"""
    if not practice_areas:
        return "Not specified"
    
    areas = []
    for delimiter in [',', ';', '\n']:
        if delimiter in practice_areas:
            areas.extend([area.strip() for area in practice_areas.split(delimiter)])
            break
    else:
        areas = [practice_areas]
    
    areas = [area for area in areas if area and not area.isspace()]
    return "\n      • " + "\n      • ".join(areas) if areas else "Not specified"

def get_claude_response(query, lawyers_summary):
    """Get Claude's analysis of the best lawyer matches"""
    summary_text = "Available Lawyers and Their Expertise:\n\n"
    for i, lawyer in enumerate(lawyers_summary, 1):
        summary_text += f"{i}. {lawyer['name']}\n"
        summary_text += f"   Availability: {lawyer['availability_status']}\n"
        summary_text += f"   Schedule: {lawyer['days_available']} days/week, {lawyer['hours_available']}/month\n"
        summary_text += f"   Practice Areas:\n{format_practice_areas(lawyer['practice_areas'])}\n"
        if lawyer['experience']:
            summary_text += f"   Industry Experience:\n{format_practice_areas(lawyer['experience'])}\n"
        summary_text += "\n"

    prompt = f"""You are a legal staffing assistant at Caravel Law. Your task is to match client needs with available lawyers based on their expertise and availability.

Client Need: {query}

{summary_text}

Please analyze the lawyers' profiles and provide the best 3-7 matches, following these guidelines:

1. Carefully evaluate each lawyer's practice areas and experience against the client's needs
2. Consider their current availability
3. Provide your recommendations in this format:
   
   Top Matches for Your Needs:

   1. [Lawyer Name]
      • Relevant Expertise: [List specific matching practice areas and experience]
      • Availability: [Include availability details]
      • Why They're a Good Fit: [Brief explanation]

   [Repeat for each recommended lawyer]

Important:
- Recommend between 3-7 lawyers, ranked by best fit
- Only include lawyers whose expertise truly matches the needs
- Be specific about why each lawyer is a good match
- If fewer than 3 strong matches exist, explain what expertise was needed but not found

Remember to focus on clear, practical matches between the client's needs and lawyers' specific expertise."""

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
        st.error(f"Error getting recommendations: {str(e)}")
        return None

def main():
    st.title("🧑‍⚖️ Caravel Law Lawyer Matcher")
    
    try:
        # Load the data files
        availability_data = pd.read_csv('Caravel Law Availability - October 18th, 2024.csv')
        bios_data = pd.read_csv('BD_Caravel.csv')
        
        # Debug checkbox in sidebar
        show_debug = st.sidebar.checkbox("Show Debug Information", False)
        
        if show_debug:
            st.sidebar.write("### Raw Data Preview")
            with st.sidebar.expander("Show Data Preview"):
                st.write("Availability Data First Few Rows:")
                st.write(availability_data.head())
                st.write("\nBios Data First Few Rows:")
                st.write(bios_data.head())
        
        lawyers_summary = prepare_lawyer_summary(availability_data, bios_data, show_debug)
        
        if not lawyers_summary:
            st.error("No available lawyers found in the system. Please check the data processing above.")
            return
        
        st.write("### How can we help you find the right lawyer?")
        st.write("Tell us about your legal needs and we'll match you with the best available lawyers.")
        
        # Example queries
        examples = [
            "I need a lawyer with trademark and IP experience who can start work soon",
            "Looking for someone experienced in employment law to help with HR policies",
            "Need a lawyer experienced with startups and financing",
            "Who would be best for drafting and negotiating SaaS agreements?"
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

        # Add spacing
        st.write("")
        
        # Three columns for filters and view all dropdown
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Get all unique practice areas
            all_practice_areas = get_practice_areas(lawyers_summary)
            selected_practice_area = st.selectbox(
                "Filter by Practice Area",
                ["All"] + all_practice_areas
            )
        
        with filter_col2:
            # Add availability filter
            availability_filter = st.selectbox(
                "Filter by Availability",
                ["All", "High Availability (3+ days/week)", 
                 "Medium Availability (1-2 days/week)", 
                 "Limited Availability (<1 day/week)"]
            )
            
        with filter_col3:
            show_all = st.expander("📊 View All Lawyers")
        
        # Add spacing
        st.write("")
        
        # Filter lawyers based on selection (this affects both view all and search)
        filtered_lawyers = lawyers_summary.copy()
        
        if selected_practice_area != "All":
            filtered_lawyers = [
                lawyer for lawyer in filtered_lawyers 
                if selected_practice_area in lawyer['practice_areas']
            ]
        
        if availability_filter != "All":
            temp_lawyers = []
            for lawyer in filtered_lawyers:
                days = lawyer['days_available']
                try:
                    days = float(days.split()[0])
                    if "High Availability" in availability_filter and days >= 3:
                        temp_lawyers.append(lawyer)
                    elif "Medium Availability" in availability_filter and 1 <= days < 3:
                        temp_lawyers.append(lawyer)
                    elif "Limited Availability" in availability_filter and days < 1:
                        temp_lawyers.append(lawyer)
                except (ValueError, AttributeError):
                    continue
            filtered_lawyers = temp_lawyers

        # Show filtered results in the expander
        with show_all:
            if filtered_lawyers:
                create_lawyer_cards(filtered_lawyers)
            else:
                st.warning("No lawyers match the selected filters.")
        
        # Custom query input for detailed search
        query = st.text_area(
            "For more specific matching, describe what you're looking for (optional):",
            value=st.session_state.get('query', ''),
            placeholder="Example: I need help with employment contracts and HR policies...",
            height=100
        )

        # Search and Clear buttons
        button_col1, button_col2 = st.columns([1, 4])
        search = button_col1.button("🔎 Search")
        clear = button_col2.button("Clear")

        if clear:
            st.session_state.query = ''
            st.rerun()

        # Show Claude's recommendations using the filtered lawyers
        if search and query:
            st.session_state.query = query
            with st.spinner("Finding the best matches..."):
                if filtered_lawyers:
                    results = get_claude_response(query, filtered_lawyers)
                    if results:
                        st.markdown("### Top Lawyer Matches")
                        st.markdown(results)
                else:
                    st.warning("No lawyers match the selected filters. Try adjusting your filters to see more options.")
            
    except FileNotFoundError as e:
        st.error("Could not find the required data files. Please check your data file locations.")
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
