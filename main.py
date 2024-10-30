import streamlit as st
import pandas as pd
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

class LawyerMatcher:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.availability_df = None
        self.bios_df = None
        self.lawyer_embeddings = None
        
    def load_data(self, availability_path, bios_path):
        """Load and prepare lawyer data"""
        self.availability_df = pd.read_csv(availability_path)
        self.bios_df = pd.read_csv(bios_path)
        
        # Create combined text for each lawyer
        self.bios_df['combined_text'] = self.bios_df.apply(
            lambda row: ' '.join(str(val) for val in row if pd.notna(val)), 
            axis=1
        )
        
        # Generate embeddings for all lawyer bios
        self.lawyer_embeddings = self.model.encode(
            self.bios_df['combined_text'].tolist(), 
            show_progress_bar=True
        )

    def rank_lawyers(self, query):
        """Rank lawyers by relevance but return all data"""
        # Generate embedding for the query
        query_embedding = self.model.encode([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_embedding, self.lawyer_embeddings)[0]
        
        # Add similarity scores to bios dataframe
        self.bios_df['relevance_score'] = similarities
        
        # Sort bios by relevance
        sorted_bios = self.bios_df.sort_values('relevance_score', ascending=False)
        
        return sorted_bios

def extract_recommended_lawyers(analysis_text, availability_df):
    """Extract recommended lawyer names and their availability info"""
    recommended_lawyers = []
    
    # Look for numbered recommendations
    pattern = r'\d+\.\s+([A-Za-z\s]+):'
    matches = re.findall(pattern, analysis_text)
    
    # Get availability info for each recommended lawyer
    for name in matches:
        name = name.strip()
        lawyer_info = availability_df[availability_df['What is your name?'].str.contains(name, case=False, na=False)]
        if not lawyer_info.empty:
            lawyer = {
                'name': name,
                'days_available': lawyer_info['What is your capacity to take on new work for the forseeable future? Days per week'].iloc[0],
                'hours_available': lawyer_info['What is your capacity to take on new work for the foreseeable future? Hours per month'].iloc[0],
                'engagement_types': lawyer_info['What type of engagement would you like to consider?'].iloc[0],
                'availability_notes': lawyer_info['Do you have any comments or instructions you should let us know about that may impact your short/long-term availability? For instance, are you going on vacation (please provide exact dates)?'].iloc[0]
            }
            recommended_lawyers.append(lawyer)
    
    return recommended_lawyers

def get_claude_response(query, availability_df, bios_df, ranked=True):
    """Get Claude's analysis with ranked data"""
    anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    prompt = f"""I have two CSV files with lawyer information. Please look at both and recommend lawyers that match this need:

{query}

First file (Availability information):
{availability_df.to_string()}

Second file (Lawyer bios and expertise - {"ranked by relevance to query" if ranked else "unranked"}):
{bios_df.to_string()}

Please analyze these files and recommend any lawyers that clearly match the need. If no lawyers have the specific expertise required, please say so directly. Be honest - only recommend lawyers if you see clear evidence they have the relevant experience.

If recommending lawyers, please format their names clearly with numbers like:
1. [Lawyer Name]: [Explanation]
2. [Lawyer Name]: [Explanation]"""

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

def chunk_dataframe(df, max_chars=8000):
    """Split dataframe into chunks to manage token count"""
    total_chars = df.to_string().count('')
    num_chunks = -(-total_chars // max_chars)  # Ceiling division
    chunk_size = len(df) // num_chunks
    return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

def main():
    st.title("Caravel Law Lawyer Matcher")
    
    # Initialize matcher
    if 'matcher' not in st.session_state:
        matcher = LawyerMatcher()
        try:
            matcher.load_data(
                'Caravel Law Availability - October 18th, 2024.csv',
                'BD_Caravel.csv'
            )
            st.session_state.matcher = matcher
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
            if col1.button(f"🔍 {example}", key=f"example_{i}"):
                st.session_state.query = example
                st.rerun()
        else:
            if col2.button(f"🔍 {example}", key=f"example_{i}"):
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
            # Rank lawyers by relevance while keeping all data
            ranked_bios = st.session_state.matcher.rank_lawyers(st.session_state.query)
            
            # If we get a rate limit error, try with chunked data
            try:
                analysis = get_claude_response(
                    st.session_state.query,
                    st.session_state.matcher.availability_df,
                    ranked_bios
                )
            except Exception as e:
                if "rate_limit_error" in str(e):
                    st.warning("Processing larger dataset in chunks...")
                    # Split data into chunks and process
                    bio_chunks = chunk_dataframe(ranked_bios)
                    availability_chunks = chunk_dataframe(st.session_state.matcher.availability_df)
                    
                    analyses = []
                    for bio_chunk, avail_chunk in zip(bio_chunks, availability_chunks):
                        chunk_analysis = get_claude_response(
                            st.session_state.query,
                            avail_chunk,
                            bio_chunk
                        )
                        if chunk_analysis:
                            analyses.append(chunk_analysis)
                    
                    analysis = "\n\n".join(analyses) if analyses else None
                else:
                    st.error(f"Error: {str(e)}")
                    analysis = None
            
        if analysis:
            # Display Claude's analysis
            st.write(analysis)
            
            # Extract and show availability for recommended lawyers
            recommended_lawyers = extract_recommended_lawyers(
                analysis, 
                st.session_state.matcher.availability_df
            )
            
            if recommended_lawyers:
                st.write("\n### Availability Details")
                for lawyer in recommended_lawyers:
                    with st.expander(f"📅 {lawyer['name']}'s Availability"):
                        st.write("**Days per Week:**", lawyer['days_available'])
                        st.write("**Hours per Month:**", lawyer['hours_available'])
                        if lawyer['engagement_types'] and str(lawyer['engagement_types']).lower() not in ['n/a', 'na', 'none', 'no']:
                            st.write("**Preferred Engagement Types:**", lawyer['engagement_types'])
                        if lawyer['availability_notes'] and str(lawyer['availability_notes']).lower() not in ['n/a', 'na', 'none', 'no']:
                            st.write("**Availability Notes:**", lawyer['availability_notes'])

if __name__ == "__main__":
    main()
