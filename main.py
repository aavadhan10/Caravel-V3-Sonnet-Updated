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

def clean_text_field(text):
    """Clean and standardize text fields"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return text.strip()

def prepare_lawyer_summary(availability_data, bios_data):
    """Create a concise summary of lawyer information"""
    # Debug prints
    st.write("### Debug Information")
    st.write("Initial data shapes:")
    st.write(f"Availability data: {availability_data.shape}")
    st.write(f"Bios data: {bios_data.shape}")
    
    # Clean and standardize name fields with debug output
    st.write("\n### Name Processing Debug")
    
    # Process availability data names
    st.write("Processing availability data names...")
    availability_data['Original_Name'] = availability_data['What is your name?'].copy()
    availability_data['Name'] = availability_data['What is your name?'].apply(lambda x: 
        ' '.join(str(x).strip().split()).title() if pd.notna(x) else '')
    
    # Process bios data names
    st.write("Processing bios data names...")
    bios_data['Original_Name'] = bios_data['First Name'] + ' ' + bios_data['Last Name']
    bios_data['Name'] = bios_data.apply(lambda x: 
        ' '.join(f"{str(x['First Name'])} {str(x['Last Name'])}".strip().split()).title() 
        if pd.notna(x['First Name']) and pd.notna(x['Last Name']) else '', axis=1)
    
    # Display name comparison
    st.write("\n### Name Matching Analysis")
    st.write("\nAvailability Data Names:")
    name_comparison = pd.DataFrame({
        'Original': availability_data['Original_Name'],
        'Processed': availability_data['Name']
    })
    st.write(name_comparison)
    
    st.write("\nBios Data Names:")
    name_comparison_bios = pd.DataFrame({
        'Original': bios_data['Original_Name'],
        'Processed': bios_data['Name']
    })
    st.write(name_comparison_bios)
    
    # Show all unique processed names
    st.write("\nUnique processed names in Availability Data:")
    st.write(sorted(availability_data['Name'].unique()))
    st.write("\nUnique processed names in Bios Data:")
    st.write(sorted(bios_data['Name'].unique()))
    
    # Find common names
    common_names = set(availability_data['Name']) & set(bios_data['Name'])
    st.write(f"\nNumber of matching names: {len(common_names)}")
    st.write("Matching names:", sorted(list(common_names)))
    
    lawyers_summary = []
    capacity_column = 'Do you have capacity to take on new work?'
    
    # Get available lawyers
    available_lawyers = availability_data[
        availability_data[capacity_column].fillna('').str.lower().str.contains('yes|maybe|y|m', na=False)
    ]
    
    st.write(f"\nNumber of available lawyers found: {len(available_lawyers)}")
    
    # Debug each lawyer matching attempt
    st.write("\n### Matching Process Debug")
    for _, row in available_lawyers.iterrows():
        name = row['Name']
        st.write(f"\nTrying to match: {name}")
        
        bio_row = bios_data[bios_data['Name'] == name]
        if bio_row.empty:
            st.write(f"No match found in bios for: {name}")
            # Try fuzzy matching or alternative name formats
            similar_names = bios_data[bios_data['Name'].str.contains(name.split()[0], na=False)]
            if not similar_names.empty:
                st.write(f"Similar names found: {similar_names['Name'].tolist()}")
        else:
            st.write(f"Match found for: {name}")
            bio_row = bio_row.iloc[0]
            
            practice_areas = clean_text_field(bio_row.get('Area of Practise + Add Info', ''))
            experience = clean_text_field(bio_row.get('Industry Experience', ''))
            
            days_available = clean_text_field(row.get(
                'What is your capacity to take on new work for the forseeable future? Days per week', 
                'Not specified'
            ))
            hours_available = clean_text_field(row.get(
                'What is your capacity to take on new work for the foreseeable future? Hours per month',
                'Not specified'
            ))
            
            summary = {
                'name': name,
                'availability_status': clean_text_field(row[capacity_column]),
                'practice_areas': practice_areas,
                'experience': experience,
                'days_available': days_available,
                'hours_available': hours_available
            }
            lawyers_summary.append(summary)
    
    st.write(f"\nFinal number of matched lawyers: {len(lawyers_summary)}")
    
    return lawyers_summary

def format_practice_areas(practice_areas):
    """Format practice areas into a bulleted list"""
    if not practice_areas:
        return "Not specified"
    
    # Split on common delimiters and clean up
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
    
    # Create a detailed but structured summary of each lawyer
    summary_text = "Available Lawyers and Their Expertise:\n\n"
    for i, lawyer in enumerate(lawyers_summary, 1):
        summary_text += f"{i}. {lawyer['name']}\n"
        summary_text += f"   Availability: {lawyer['availability_status']}\n"
        summary_text += f"   Schedule: {lawyer['days_available']} days/week, {lawyer['hours_available']}/month\n"
        summary_text += f"   Practice Areas:\n{format_practice_areas(lawyer['practice_areas'])}\n"
        if lawyer['experience']:
            summary_text += f"   Industry Experience:\n{format_practice_areas(lawyer['experience'])}\n"
        summary_text += "\n"

    # Debug: Print the summary text
    st.write("### Generated Lawyer Summary")
    st.write(summary_text)

    prompt = f"""You are a legal staffing assistant at Caravel Law. Your task is to match client needs with available lawyers based on their expertise and availability.

Client Need: {query}

{summary_text}

Please analyze the lawyers' profiles and provide recommendations based on the following criteria:
1. Match the client's specific needs with lawyers' practice areas and experience
2. Consider the lawyers' current availability
3. If you find suitable matches:
   - Recommend up to 3 lawyers, ranked by how well they match the requirements
   - Explain specifically why each lawyer is a good match, citing their relevant expertise
   - Include their availability details
4. If no exact matches are found but there are lawyers with related expertise:
   - Suggest these lawyers as potential alternatives
   - Explain how their expertise might transfer to the client's needs
5. If no suitable matches are found:
   - Clearly explain what specific expertise was needed but not found
   - Recommend next steps for the client

Please be thorough in your analysis and specific in your recommendations."""

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
        st.error(f"Claude API error: {str(e)}")
        return None

def main():
    st.title("Caravel Law Lawyer Matcher")
    
    try:
        # Load the data files with explicit display of contents
        st.write("### Loading Data Files")
        
        availability_data = pd.read_csv('Caravel Law Availability - October 18th, 2024.csv')
        st.write("Successfully loaded availability data")
        
        bios_data = pd.read_csv('BD_Caravel.csv')
        st.write("Successfully loaded bios data")
        
        # Show full data preview
        st.write("### Raw Data Preview")
        st.write("Availability Data First Few Rows:")
        st.write(availability_data.head())
        st.write("\nBios Data First Few Rows:")
        st.write(bios_data.head())
        
        lawyers_summary = prepare_lawyer_summary(availability_data, bios_data)
        
        if not lawyers_summary:
            st.error("No available lawyers found in the system. Please check the data processing above.")
            return
        
        # Example queries
        st.write("### How can we help you find the right lawyer?")
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
            "Or describe what you're looking for:",
            value=st.session_state.get('query', ''),
            placeholder="Describe your legal needs...",
            height=100
        )

        # Search and Clear buttons
        col1, col2 = st.columns([1, 4])
        search = col1.button("Find Lawyers")
        clear = col2.button("Clear")

        if clear:
            st.session_state.query = ''
            st.rerun()

        if search and query:
            st.session_state.query = query
            with st.spinner("Finding the best matches..."):
                results = get_claude_response(query, lawyers_summary)
                if results:
                    st.markdown("### Recommendations")
                    st.markdown(results)
            
    except FileNotFoundError as e:
        st.error(f"Could not find one or more data files: {str(e)}")
        st.write("Please ensure the following files exist in the correct location:")
        st.write("1. 'Caravel Law Availability - October 18th, 2024.csv'")
        st.write("2. 'BD_Caravel.csv'")
        return
    except Exception as e:
        st.error(f"Error processing data files: {str(e)}")
        st.write("Error details:", str(e))
        return

if __name__ == "__main__":
    main()
