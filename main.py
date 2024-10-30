import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from anthropic import Anthropic
import os

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def load_data():
    # Load availability data
    availability_df = pd.read_csv('Caravel Law Availability - October 18th, 2024.csv')
    
    # Load bio data
    bios_df = pd.read_csv('BD_Caravel.csv')
    
    # Clean up lawyer names in availability data
    availability_df['Name'] = availability_df['What is your name?'].str.strip()
    
    # Create full name in bios data
    bios_df['Name'] = bios_df['First Name'].str.strip() + ' ' + bios_df['Last Name'].str.strip()
    
    # Merge the dataframes
    merged_df = pd.merge(
        availability_df,
        bios_df,
        on='Name',
        how='inner'
    )
    
    return merged_df

def get_claude_analysis(query, lawyer_info):
    """Get Claude's analysis of how well a lawyer matches the query."""
    prompt = f"""As an expert legal staffing professional, analyze how well this lawyer matches the client's needs. Consider both explicit requirements and implicit needs based on the context.

CLIENT NEED:
{query}

LAWYER PROFILE:
- Name: {lawyer_info.get('Name', '')}
- Call Year: {lawyer_info.get('Call', '')}
- Practice Areas: {lawyer_info.get('Area of Practise + Add Info', '')}
- Industry Experience: {lawyer_info.get('Industry Experience', '')}
- Notable Experience: {lawyer_info.get('Notable Items/Personal Details', '')}
- Location: {lawyer_info.get('Location', '')}
- Languages: {lawyer_info.get('Languages', '')}
- Previous Companies: {lawyer_info.get('Previous Companies/Firms', '')}

Please evaluate the match considering:
1. Direct expertise match with client needs
2. Relevant industry experience
3. Years of experience and seniority level
4. Complementary skills that could add value
5. Any potential gaps or limitations

Provide your response in the following format:
Score: [0-100]
Primary Strengths: [Key reasons this lawyer is a good match]
Potential Limitations: [Any areas where the lawyer might not fully meet the needs]
Overall Recommendation: [Brief synthesis of whether and why this is a good match]
"""
    
    try:
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Parse response
        content = response.content[0].text
        score = int([line for line in content.split('\n') if line.startswith('Score:')][0].split(':')[1].strip())
        strengths = [line for line in content.split('\n') if line.startswith('Primary Strengths:')][0].split(':')[1].strip()
        limitations = [line for line in content.split('\n') if line.startswith('Potential Limitations:')][0].split(':')[1].strip()
        recommendation = [line for line in content.split('\n') if line.startswith('Overall Recommendation:')][0].split(':')[1].strip()
        
        return {
            'score': score,
            'strengths': strengths,
            'limitations': limitations,
            'recommendation': recommendation
        }
    except Exception as e:
        st.error(f"Error getting Claude's analysis: {e}")
        return {
            'score': 0,
            'strengths': '',
            'limitations': 'Error in analysis',
            'recommendation': ''
        }

def get_alternative_recommendations(query, top_matches, all_lawyers):
    """Get Claude's suggestions for alternative lawyers to consider."""
    prompt = f"""As an expert legal staffing professional, review these top matching lawyers and suggest alternative options that might bring different strengths to the client's needs.

CLIENT NEED:
{query}

TOP MATCHING LAWYERS:
{[f"{lawyer['Name']}: {lawyer['Area of Practise + Add Info']}" for lawyer in top_matches[:3]]}

Consider recommending lawyers who:
1. Bring complementary expertise
2. Offer different industry perspectives
3. Have unique combinations of skills
4. Might approach the problem differently

Analyze the following alternatives and suggest up to 3 that would be worth considering:

{[f"{lawyer['Name']}: {lawyer['Area of Practise + Add Info']}" for lawyer in all_lawyers[:10] if lawyer['Name'] not in [m['Name'] for m in top_matches[:3]]]}

Format your response as:
Alternative 1: [Name] - [Reason for recommendation]
Alternative 2: [Name] - [Reason for recommendation]
Alternative 3: [Name] - [Reason for recommendation]
"""

    try:
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        return response.content[0].text
    except Exception as e:
        st.error(f"Error getting alternative recommendations: {e}")
        return "Unable to generate alternative recommendations."

def search_lawyers(df, search_query, min_days=1):
    # Convert days to numeric, taking the first value if multiple are provided
    df['Days'] = df['What is your capacity to take on new work for the forseeable future? Days per week'].str.split(';').str[0].str.extract('(\d+)').astype(float)
    
    # Filter for available lawyers
    available = df[
        (df['Do you have capacity to take on new work?'].str.lower() == 'yes') & 
        (df['Days'] >= min_days)
    ].copy()
    
    if not available.empty and search_query:
        # Get Claude's analysis for each lawyer
        results = []
        for _, lawyer in available.iterrows():
            analysis = get_claude_analysis(search_query, lawyer)
            lawyer_result = lawyer.to_dict()
            lawyer_result.update(analysis)
            results.append(lawyer_result)
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Get alternative recommendations for top matches
        if len(results) >= 3:
            alternatives = get_alternative_recommendations(search_query, results[:3], results[3:])
        else:
            alternatives = None
        
        return results, alternatives
    
    return [], None

def main():
    st.title("Caravel Law Lawyer Matcher")
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        st.error("Please set the ANTHROPIC_API_KEY environment variable")
        return
    
    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Create sidebar for filters
    st.sidebar.header("Search Filters")
    
    # Search interface
    search_query = st.sidebar.text_area(
        "Describe what you're looking for",
        placeholder="E.g., 'I need a lawyer experienced in technology licensing and privacy law for a SaaS company. The project involves international data protection compliance and contract negotiation.'"
    )
    
    min_days = st.sidebar.slider(
        "Minimum days per week required",
        min_value=1,
        max_value=5,
        value=1
    )
    
    if st.sidebar.button("Search"):
        if search_query:
            with st.spinner("Analyzing lawyer profiles..."):
                results, alternatives = search_lawyers(df, search_query, min_days)
            
            if results:
                st.write("### Top Matches")
                
                for lawyer in results[:5]:  # Show top 5 matches
                    with st.expander(f"{lawyer['Name']} - Match Score: {lawyer['score']}%"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write("**Match Analysis:**")
                            st.write("🎯 **Primary Strengths:**", lawyer['strengths'])
                            if lawyer['limitations']:
                                st.write("⚠️ **Potential Limitations:**", lawyer['limitations'])
                            st.write("📋 **Recommendation:**", lawyer['recommendation'])
                        
                        with col2:
                            st.write("**Availability:**")
                            st.write(f"- Days per week: {lawyer['Days']}")
                            st.write(f"- Preferred engagement: {lawyer['What type of engagement would you like to consider?']}")
                
                if alternatives:
                    st.write("### Alternative Recommendations")
                    st.write("Consider these lawyers who might bring different perspectives:")
                    st.write(alternatives)
                
            else:
                st.warning("No lawyers found matching your criteria.")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()
