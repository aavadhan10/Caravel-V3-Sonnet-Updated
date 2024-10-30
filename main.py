import streamlit as st
import pandas as pd
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import re
import time
from typing import List, Dict
import torch

# Load environment variables
load_dotenv()

class LawyerMatcher:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error initializing SentenceTransformer: {str(e)}")
            self.model = None
        self.availability_df = None
        self.bios_df = None
        self.lawyer_embeddings = None
        
    def load_data(self, availability_path: str, bios_path: str):
        """Load and prepare lawyer data"""
        try:
            self.availability_df = pd.read_csv(availability_path)
            self.bios_df = pd.read_csv(bios_path, delimiter='\t')  # Using tab delimiter for BD_Caravel.csv
            
            # Create full name column in bios_df
            self.bios_df['full_name'] = self.bios_df['First Name'] + ' ' + self.bios_df['Last Name']
            
            # Create combined text for each lawyer's bio
            relevant_columns = [
                'Level/Title', 'Call', 'Jurisdiction', 'Location', 
                'Area of Practise + Add Info', 'Industry Experience', 
                'Languages', 'Previous In-House Companies', 
                'Previous Companies/Firms', 'Education', 
                'Awards/Recognition', 'Notable Items/Personal Details'
            ]
            
            self.bios_df['combined_text'] = self.bios_df.apply(
                lambda row: ' '.join(str(val) for col in relevant_columns 
                                   if col in row.index and pd.notna(row[col]) 
                                   for val in str(row[col]).split(';')), 
                axis=1
            )
            
            # Generate embeddings for all lawyer bios if model is available
            if self.model is not None:
                self.lawyer_embeddings = self.model.encode(
                    self.bios_df['combined_text'].tolist(), 
                    show_progress_bar=True
                )
            else:
                st.warning("SentenceTransformer model not available. Falling back to basic text matching.")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            raise

    def rank_lawyers(self, query: str) -> pd.DataFrame:
        """Rank lawyers by relevance to query"""
        if self.model is not None and self.lawyer_embeddings is not None:
            query_embedding = self.model.encode([query])
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, self.lawyer_embeddings)[0]
            self.bios_df['relevance_score'] = similarities
        else:
            # Fallback to basic text matching
            self.bios_df['relevance_score'] = self.bios_df['combined_text'].apply(
                lambda x: sum(q.lower() in x.lower() for q in query.split())
            )
        return self.bios_df.sort_values('relevance_score', ascending=False)

def chunk_data(bios_df: pd.DataFrame, availability_df: pd.DataFrame, chunk_size: int = 5) -> List[Dict[str, pd.DataFrame]]:
    """Split data into smaller chunks"""
    chunks = []
    
    try:
        # Get common lawyer names between both dataframes using full name
        bios_full_names = bios_df['full_name'].tolist()
        availability_names = availability_df['What is your name?'].tolist()
        
        # Create a mapping of standardized names
        def standardize_name(name):
            return ' '.join(sorted(name.lower().split()))
        
        bios_std_names = {standardize_name(name): name for name in bios_full_names}
        avail_std_names = {standardize_name(name): name for name in availability_names}
        
        # Find common lawyers
        common_std_names = set(bios_std_names.keys()) & set(avail_std_names.keys())
        
        if not common_std_names:
            raise ValueError("No matching lawyers found between bios and availability data")
        
        # Filter dataframes to only include common lawyers
        bios_df = bios_df[bios_df['full_name'].apply(standardize_name).isin(common_std_names)]
        availability_df = availability_df[availability_df['What is your name?'].apply(standardize_name).isin(common_std_names)]
        
        # Create chunks
        for i in range(0, len(bios_df), chunk_size):
            chunk_bios = bios_df.iloc[i:i + chunk_size]
            chunk_lawyers = chunk_bios['full_name'].apply(standardize_name).tolist()
            chunk_availability = availability_df[
                availability_df['What is your name?'].apply(standardize_name).isin(chunk_lawyers)
            ]
            
            chunks.append({
                'bios': chunk_bios,
                'availability': chunk_availability
            })
        
        return chunks
    except Exception as e:
        st.error(f"Error chunking data: {str(e)}")
        return []

def extract_recommended_lawyers(analysis_text: str, availability_df: pd.DataFrame) -> List[Dict]:
    """Extract recommended lawyer names and their availability info"""
    recommended_lawyers = []
    pattern = r'\d+\.\s+([A-Za-z\s]+):'
    matches = re.findall(pattern, analysis_text)
    
    for name in matches:
        name = name.strip()
        # Try to find the lawyer by standardized name
        std_name = ' '.join(sorted(name.lower().split()))
        lawyer_info = availability_df[
            availability_df['What is your name?'].apply(lambda x: ' '.join(sorted(x.lower().split()))) == std_name
        ]
        
        if not lawyer_info.empty:
            try:
                lawyer = {
                    'name': name,
                    'days_available': lawyer_info['Daily/Fractional Engagements'].iloc[0],
                    'hours_available': lawyer_info['Monthly Engagements (hours per month)'].iloc[0],
                    'upcoming_vacation': lawyer_info['Upcoming Vacation'].iloc[0] if 'Upcoming Vacation' in lawyer_info.columns else 'Not specified'
                }
                recommended_lawyers.append(lawyer)
            except Exception as e:
                st.warning(f"Error extracting availability info for {name}: {str(e)}")
    
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
            st.error(f"Error initializing application: {str(e)}")
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
            ranked_bios = st.session_state.matcher.rank_lawyers(st.session_state.query)
            chunks = chunk_data(ranked_bios, st.session_state.matcher.availability_df)
            
            all_analyses = []
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                chunk_analysis = process_chunk(st.session_state.query, chunk)
                if chunk_analysis and any(re.findall(r'\d+\.\s+([A-Za-z\s]+):', chunk_analysis)):
                    all_analyses.append(chunk_analysis)
                progress_bar.progress((i + 1) / len(chunks))
            
            if all_analyses:
                analysis = "\n\n".join(all_analyses)
                st.write(analysis)
                
                recommended_lawyers = extract_recommended_lawyers(
                    analysis, 
                    st.session_state.matcher.availability_df
                )
                
                if recommended_lawyers:
                    st.write("\n### Availability Details")
                    for lawyer in recommended_lawyers:
                        with st.expander(f"📅 {lawyer['name']}'s Availability"):
                            if pd.notna(lawyer['days_available']):
                                st.write("**Days Available:**", lawyer['days_available'])
                            if pd.notna(lawyer['hours_available']):
                                st.write("**Hours per Month:**", lawyer['hours_available'])
                            if pd.notna(lawyer['upcoming_vacation']) and str(lawyer['upcoming_vacation']).lower() not in ['n/a', 'na', 'none', 'no']:
                                st.write("**Upcoming Vacation:**", lawyer['upcoming_vacation'])
                else:
                    st.warning("No matching lawyers found.")

if __name__ == "__main__":
    main()
