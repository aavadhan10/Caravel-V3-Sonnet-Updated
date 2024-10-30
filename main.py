import streamlit as st
import pandas as pd
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from typing import List, Dict

# Load environment variables
load_dotenv()

def retry_with_exponential_backoff(func):
    """Decorator for retrying functions with exponential backoff"""
    def wrapper(*args, **kwargs):
        max_retries = 5
        retry_delay = 1  # Initial delay in seconds
        
        for retry in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate_limit_error" in str(e):
                    if retry < max_retries - 1:
                        delay = retry_delay * (2 ** retry)  # Exponential backoff
                        st.warning(f"Rate limit reached. Waiting {delay} seconds before retry...")
                        time.sleep(delay)
                        continue
                raise e
    return wrapper

class LawyerMatcher:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.availability_df = None
        self.bios_df = None
        self.lawyer_embeddings = None
        
    def load_data(self, availability_path: str, bios_path: str):
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

    def rank_lawyers(self, query: str) -> pd.DataFrame:
        """Rank lawyers by relevance to query"""
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.lawyer_embeddings)[0]
        self.bios_df['relevance_score'] = similarities
        return self.bios_df.sort_values('relevance_score', ascending=False)

def chunk_data(bios_df: pd.DataFrame, availability_df: pd.DataFrame, chunk_size: int = 5) -> List[Dict[str, pd.DataFrame]]:
    """Split data into smaller chunks"""
    chunks = []
    
    # Sort both dataframes by the same lawyer names to maintain alignment
    lawyer_names = bios_df['What is your name?'].tolist()
    availability_df = availability_df[availability_df['What is your name?'].isin(lawyer_names)]
    
    # Create chunks
    for i in range(0, len(bios_df), chunk_size):
        chunk_bios = bios_df.iloc[i:i + chunk_size]
        chunk_lawyers = chunk_bios['What is your name?'].tolist()
        chunk_availability = availability_df[availability_df['What is your name?'].isin(chunk_lawyers)]
        
        chunks.append({
            'bios': chunk_bios,
            'availability': chunk_availability
        })
    
    return chunks

@retry_with_exponential_backoff
def process_chunk(query: str, chunk: Dict[str, pd.DataFrame]) -> str:
    """Process a single chunk of data with Claude"""
    anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    prompt = f"""I have information about a subset of lawyers. Please analyze if any match this need:

{query}

Availability information:
{chunk['availability'].to_string()}

Lawyer bios and expertise:
{chunk['bios'].to_string()}

Please recommend any lawyers from this group that clearly match the need. If none have the specific expertise required, please say so directly. Be honest - only recommend lawyers if you see clear evidence they have the relevant experience.

If recommending lawyers, please format their names clearly with numbers like:
1. [Lawyer Name]: [Explanation]
2. [Lawyer Name]: [Explanation]"""

    response = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    return response.content[0].text

def extract_recommended_lawyers(analysis_text: str, availability_df: pd.DataFrame) -> List[Dict]:
    """Extract recommended lawyer names and their availability info"""
    recommended_lawyers = []
    pattern = r'\d+\.\s+([A-Za-z\s]+):'
    matches = re.findall(pattern, analysis_text)
    
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
            # Rank lawyers by relevance
            ranked_bios = st.session_state.matcher.rank_lawyers(st.session_state.query)
            
            # Split data into chunks
            chunks = chunk_data(ranked_bios, st.session_state.matcher.availability_df)
            
            # Process each chunk
            all_analyses = []
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                chunk_analysis = process_chunk(st.session_state.query, chunk)
                if chunk_analysis and any(re.findall(r'\d+\.\s+([A-Za-z\s]+):', chunk_analysis)):
                    all_analyses.append(chunk_analysis)
                progress_bar.progress((i + 1) / len(chunks))
            
            # Combine results
            if all_analyses:
                analysis = "\n\n".join(all_analyses)
                
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
            else:
                st.warning("No matching lawyers found.")

if __name__ == "__main__":
    main()
