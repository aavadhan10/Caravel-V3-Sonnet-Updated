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

def prepare_lawyer_summary(availability_data, bios_data):
    """Create a concise summary of lawyer information"""
    # Log the initial data
    logger.info(f"Number of lawyers in availability data: {len(availability_data)}")
    logger.info(f"Number of lawyers in bios data: {len(bios_data)}")
    
    # Merge the dataframes
    availability_data['Name'] = availability_data['What is your name?'].str.strip()
    bios_data['Name'] = bios_data['First Name'].str.strip() + ' ' + bios_data['Last Name'].str.strip()
    
    lawyers_summary = []
    
    # Get available lawyers
    available_lawyers = availability_data[
        availability_data['Do you have capacity to take on new work?'].str.lower().isin(['yes', 'maybe'])
    ]
    
    logger.info(f"Number of available lawyers: {len(available_lawyers)}")
    
    for _, row in available_lawyers.iterrows():
        name = row['Name']
        bio_row = bios_data[bios_data['Name'] == name]
        
        if not bio_row.empty:
            bio_row = bio_row.iloc[0]
            practice_areas = bio_row.get('Area of Practise + Add Info', '').strip()
            experience = bio_row.get('Industry Experience', '').strip()
            
            # Log each lawyer's info for debugging
            logger.info(f"\nLawyer: {name}")
            logger.info(f"Practice Areas: {practice_areas}")
            logger.info(f"Experience: {experience}")
            
            summary = {
                'name': name,
                'availability': row['What is your capacity to take on new work?'],
                'practice_areas': practice_areas,
                'experience': experience,
                'days_available': row.get('What is your capacity to take on new work for the forseeable future? Days per week', ''),
                'hours_available': row.get('What is your capacity to take on new work for the foreseeable future? Hours per month', '')
            }
            lawyers_summary.append(summary)
    
    return lawyers_summary

def format_practice_areas(practice_areas):
    """Format practice areas into a bulleted list"""
    if not practice_areas:
        return "Not specified"
    
    # Split on common delimiters and clean up
    areas = [area.strip() for area in practice_areas.replace(';', ',').split(',')]
    areas = [area for area in areas if area]
    
    if not areas:
        return "Not specified"
    
    return "\n      • " + "\n      • ".join(areas)

def get_claude_response(query, lawyers_summary):
    """Get Claude's analysis of the best lawyer matches"""
    
    # Debug: Print total number of lawyers being processed
    logger.info(f"Processing {len(lawyers_summary)} lawyers for matching")
    
    # Create a numbered list of available lawyers with better formatting
    summary_text = "Here are the ONLY lawyers you can choose from:\n"
    for i, lawyer in enumerate(lawyers_summary, 1):
        summary_text += f"\n{i}. {lawyer['name']}:"
        summary_text += f"\n   Practice Areas: {format_practice_areas(lawyer['practice_areas'])}"
        if lawyer['experience']:
            summary_text += f"\n   Industry Experience: {lawyer['experience']}"
        summary_text += f"\n   Availability: {lawyer['days_available']} days/week, {lawyer['hours_available']}/month\n"
    
    # Log the formatted summary for debugging
    logger.info("\nFormatted Lawyer Summary:")
    logger.info(summary_text)
    
    # Create a list of valid lawyer names for validation
    valid_names = [lawyer['name'] for lawyer in lawyers_summary]
    valid_names_str = ", ".join(valid_names)
    
    prompt = f"""You are a legal staffing assistant. Your task is to match client needs with available lawyers.

IMPORTANT: You must ONLY recommend lawyers from this exact list: {valid_names_str}
Do not suggest any lawyers not in this list. If no lawyers match the requirements, say so directly.

Client Need: {query}

{summary_text}

Instructions:
1. Carefully review each lawyer's practice areas and experience
2. Look for specific mentions of expertise related to the client's needs
3. If matches are found:
   - Recommend 2-3 lawyers from the above list, ranked by fit
   - For each recommendation, cite specific practice areas or experience that match the client's needs
   - Include their availability details
4. If no matches are found:
   - Clearly state that no lawyers in the current list have the required expertise
   - Be specific about what expertise was looked for but not found

Remember: Only recommend lawyers if their listed practice areas or experience explicitly match the client's needs."""

    try:
        # Log the prompt for debugging
        logger.info("\nSending prompt to Claude:")
        logger.info(prompt)
        
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Log Claude's response for debugging
        logger.info("\nClaude's response:")
        logger.info(response.content[0].text)
        
        return response.content[0].text
        
    except Exception as e:
        if 'rate_limit_error' in str(e):
            st.warning("Rate limit reached. Waiting a moment before trying again...")
            time.sleep(5)
            try:
                response = anthropic.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                return response.content[0].text
            except Exception as e2:
                st.error("Still unable to get recommendations. Please try again in a moment.")
                return None
        st.error(f"Error getting recommendations: {str(e)}")
        return None

def main():
    st.title("Caravel Law Lawyer Matcher")
    
    try:
        # Load the data files
        availability_data = pd.read_csv('Caravel Law Availability - October 18th, 2024.csv')
        bios_data = pd.read_csv('BD_Caravel.csv')
        
        # Add debug output to see data structure
        if st.checkbox("Show Debug Info"):
            st.write("### Data Preview")
            st.write("Availability Data Columns:", availability_data.columns.tolist())
            st.write("Bios Data Columns:", bios_data.columns.tolist())
            st.write("\nSample Bio Data:")
            st.write(bios_data[['First Name', 'Last Name', 'Area of Practise + Add Info', 'Industry Experience']].head())
        
        # Prepare summary once at startup
        lawyers_summary = prepare_lawyer_summary(availability_data, bios_data)
        
    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
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
        placeholder="Describe your legal needs..."
    )

    # Search and Clear buttons
    col1, col2 = st.columns([1, 4])
    search = col1.button("Find Lawyers")
    clear = col2.button("Clear")

    if clear:
        st.session_state.query = ''
        st.rerun()

    if search:
        st.session_state.query = query
        st.rerun()

    # Process search
    if st.session_state.get('query'):
        with st.spinner("Finding the best matches..."):
            results = get_claude_response(st.session_state.query, lawyers_summary)
            
        if results:
            st.write(results)

if __name__ == "__main__":
    main()
