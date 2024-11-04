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
    
    # Display column names
    st.write("\nAvailability data columns:")
    st.write(availability_data.columns.tolist())
    st.write("\nBios data columns:")
    st.write(bios_data.columns.tolist())
    
    # Debug: Show unique values in capacity column
    capacity_column = 'Do you have capacity to take on new work?'
    if capacity_column in availability_data.columns:
        st.write("\nUnique values in capacity column:")
        st.write(availability_data[capacity_column].unique())
    
    # Clean and standardize name fields
    availability_data['Name'] = availability_data['What is your name?'].apply(clean_text_field)
    bios_data['Name'] = bios_data.apply(lambda x: 
        f"{clean_text_field(x['First Name'])} {clean_text_field(x['Last Name'])}", axis=1)
    
    # Debug: Show some sample names
    st.write("\nSample names from availability data:")
    st.write(availability_data['Name'].head())
    st.write("\nSample names from bios data:")
    st.write(bios_data['Name'].head())
    
    lawyers_summary = []
    
    # Get available lawyers with more lenient filtering
    available_lawyers = availability_data[
        availability_data[capacity_column].fillna('').str.lower().str.contains('yes|maybe|y|m', na=False)
    ]
    
    st.write(f"\nNumber of available lawyers found: {len(available_lawyers)}")
    
    for _, row in available_lawyers.iterrows():
        name = row['Name']
        bio_row = bios_data[bios_data['Name'] == name]
        
        if not bio_row.empty:
            bio_row = bio_row.iloc[0]
            
            # Clean and process practice areas and experience
            practice_areas = clean_text_field(bio_row.get('Area of Practise + Add Info', ''))
            experience = clean_text_field(bio_row.get('Industry Experience', ''))
            
            # Get availability details with proper error handling
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
