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
    
    # Convert to string and clean
    name = str(name).lower().strip()
    
    # Remove parenthetical contents
    name = ' '.join([part for part in name.split() if '(' not in part and ')' not in part])
    
    # Remove middle names (keep first and last only)
    name_parts = name.split()
    if len(name_parts) > 2:
        name = f"{name_parts[0]} {name_parts[-1]}"
    
    # Standardize hyphens
    name = name.replace('- ', '-').replace(' -', '-')
    
    # Remove 's' from end of last name if present
    last_name = name.split()[-1]
    if last_name.endswith('s'):
        name = f"{' '.join(name.split()[:-1])} {last_name[:-1]}"
    
    return name

def clean_text_field(text):
    """Clean and standardize text fields"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return text.strip()

def prepare_lawyer_summary(availability_data, bios_data, show_debug=False):
    """Create a concise summary of lawyer information"""
    if show_debug:
        st.write("### Debug Information")
        st.write("Initial data shapes:")
        st.write(f"Availability data: {availability_data.shape}")
        st.write(f"Bios data: {bios_data.shape}")
    
    # Process availability data names
    availability_data['Original_Name'] = availability_data['What is your name?'].copy()
    availability_data['Standardized_Name'] = availability_data['What is your name?'].apply(standardize_name)
    
    # Process bios data names
    bios_data['Original_Name'] = bios_data['First Name'] + ' ' + bios_data['Last Name']
    bios_data['Standardized_Name'] = bios_data.apply(
        lambda x: standardize_name(f"{x['First Name']} {x['Last Name']}"), axis=1)
    
    if show_debug:
        st.write("\n### Name Standardization Results")
        st.write("\nAvailability Data Names:")
        name_comparison = pd.DataFrame({
            'Original': availability_data['Original_Name'],
            'Standardized': availability_data['Standardized_Name']
        })
        st.write(name_comparison)
        
        st.write("\nBios Data Names:")
        name_comparison_bios = pd.DataFrame({
            'Original': bios_data['Original_Name'],
            'Standardized': bios_data['Standardized_Name']
        })
        st.write(name_comparison_bios)
    
    lawyers_summary = []
    capacity_column = 'Do you have capacity to take on new work?'
    
    # Get available lawyers
    available_lawyers = availability_data[
        availability_data[capacity_column].fillna('').str.lower().str.contains('yes|maybe|y|m', na=False)
    ]
    
    if show_debug:
        st.write(f"\nNumber of available lawyers found: {len(available_lawyers)}")
    
    # Match lawyers using standardized names
    for _, row in available_lawyers.iterrows():
        std_name = row['Standardized_Name']
        bio_row = bios_data[bios_data['Standardized_Name'] == std_name]
        
        if not bio_row.empty:
            if show_debug:
                st.write(f"Match found: {row['Original_Name']} ↔ {bio_row.iloc[0]['Original_Name']}")
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
        st.write(f"\nFinal number of matched lawyers: {len(lawyers_summary)}")
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
            st.write("### Raw Data Preview")
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

        # Custom query input
        query = st.text_area(
            "Describe what you're looking for:",
            value=st.session_state.get('query', ''),
            placeholder="Example: I need help with employment contracts and HR policies...",
            height=100
        )

        # Search and Clear buttons
        col1, col2 = st.columns([1, 4])
        search = col1.button("🔎 Search")
        clear = col2.button("Clear")

        if clear:
            st.session_state.query = ''
            st.rerun()

        if search and query:
            st.session_state.query = query
            with st.spinner("Finding the best matches..."):
                results = get_claude_response(query, lawyers_summary)
                if results:
                    st.markdown("### Top Lawyer Matches")
                    st.markdown(results)
            
    except FileNotFoundError as e:
        st.error("Could not find the required data files. Please check your data file locations.")
        if show_debug:
            st.write("Error details:", str(e))
        return
    except Exception as e:
        st.error("An error occurred while processing the data.")
        if show_debug:
            st.write("Error details:", str(e))
        return

if __name__ == "__main__":
    main()
