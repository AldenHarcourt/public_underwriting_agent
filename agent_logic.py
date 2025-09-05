"""
Database-integrated AI Underwriting Agent Logic
Simplified version that uses Supabase database instead of CSV files
"""

import sys
import time
from geopy.geocoders import ArcGIS
import random
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import openai
import requests
import re
import json
import ast
from io import StringIO
import os
import streamlit as st
import pydeck as pdk
import ast
import urllib.parse
from geopy.exc import GeocoderTimedOut

# Import our REST database client (avoids supabase library dependency issues)
from database_client_rest import load_data_from_database, find_comps_from_database, get_rest_database_client

# --- Agent Configuration ---
# OpenAI disabled for testing
# OpenAI API Key Configuration
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key", "")
except:
    import os
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize OpenAI client
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

# --- Constants ---
MINIMUM_COMPS = 3
RESIDENTIAL_PROPERTY_TYPES = ['singleFamily', 'multiFamily', 'condo', 'townhome', 'apartment', 'manufactured']

# --- Utility Functions ---

def safe_to_numeric(series_or_value, column_name=""):
    """Safely convert to numeric with consistent error handling."""
    return pd.to_numeric(series_or_value, errors='coerce')

def safe_to_datetime(series_or_value, column_name=""):
    """Safely convert to datetime with consistent error handling."""
    return pd.to_datetime(series_or_value, errors='coerce')

def get_property_type(property_dict, default='singleFamily'):
    """Get property type from property dictionary with consistent default."""
    return property_dict.get('propertyType', default)

def calculate_comp_scores(comps_df, subject_property):
    """
    Calculate composite scores for comparables based on proximity, recency, and similarity.
    Higher scores indicate better comparables that should have more weight in valuation.
    """
    import numpy as np
    
    subject_beds = subject_property.get('bedrooms', 0)
    subject_baths = subject_property.get('bathrooms', 0)
    subject_sqft = subject_property.get('sqft', 0)
    
    scores = []
    
    for _, comp in comps_df.iterrows():
        score = 1.0  # Base score
        
        # Distance scoring (closer = better)
        distance = comp.get('distance_miles', 10)
        if distance <= 0.25:
            distance_score = 1.0
        elif distance <= 0.5:
            distance_score = 0.9
        elif distance <= 1.0:
            distance_score = 0.8
        elif distance <= 2.0:
            distance_score = 0.6
        else:
            distance_score = 0.4
        
        # Recency scoring (more recent = better)
        months_old = comp.get('months_since_sale', 36)
        if months_old <= 6:
            recency_score = 1.0
        elif months_old <= 12:
            recency_score = 0.9
        elif months_old <= 18:
            recency_score = 0.8
        elif months_old <= 24:
            recency_score = 0.7
        else:
            recency_score = 0.5
        
        # Bed/bath similarity scoring
        bed_diff = abs(comp.get('bedrooms', 0) - subject_beds)
        bath_diff = abs(comp.get('bathrooms', 0) - subject_baths)
        
        if bed_diff == 0 and bath_diff <= 0.5:
            bed_bath_score = 1.0
        elif bed_diff <= 1 and bath_diff <= 1:
            bed_bath_score = 0.9
        elif bed_diff <= 1 and bath_diff <= 1.5:
            bed_bath_score = 0.8
        elif bed_diff <= 2 and bath_diff <= 2:
            bed_bath_score = 0.6
        else:
            bed_bath_score = 0.4
        
        # Square footage similarity scoring
        if subject_sqft > 0:
            comp_sqft = comp.get('sqft', 0)
            if comp_sqft > 0:
                sqft_ratio = min(comp_sqft, subject_sqft) / max(comp_sqft, subject_sqft)
                if sqft_ratio >= 0.9:
                    sqft_score = 1.0
                elif sqft_ratio >= 0.8:
                    sqft_score = 0.9
                elif sqft_ratio >= 0.7:
                    sqft_score = 0.8
                elif sqft_ratio >= 0.6:
                    sqft_score = 0.6
                else:
                    sqft_score = 0.4
            else:
                sqft_score = 0.5  # Missing sqft data
        else:
            sqft_score = 1.0  # No subject sqft to compare
        
        # Composite score with weighted factors
        composite_score = (
            distance_score * 0.35 +    # Distance is most important
            recency_score * 0.30 +     # Recency is very important  
            bed_bath_score * 0.20 +    # Bed/bath similarity important
            sqft_score * 0.15          # Sqft similarity moderately important
        )
        
        scores.append(composite_score)
    
    return pd.Series(scores, index=comps_df.index)

# Lot size adjustment functions removed due to reliability concerns
# Focus on weighted comp scoring for accurate valuations

def calculate_months_since_sale(comps_df, today=None):
    """Calculate months since sale for a DataFrame of comps."""
    if today is None:
        today = pd.Timestamp.today().normalize()
    return comps_df['dateSold'].apply(
        lambda d: (today.year - d.year) * 12 + (today.month - d.month) if pd.notnull(d) else 999
    )

def calculate_distance_miles(comps_df, subject_coords):
    """Calculate distance in miles from subject property to each comp."""
    return comps_df.apply(
        lambda row: geodesic((row['latitude'], row['longitude']), subject_coords).miles, 
        axis=1
    )

def extract_lot_size_in_acres(property_dict):
    """Extract lot size in acres with unit conversion."""
    lot_size = property_dict.get('lotSize', 0)
    lot_unit = property_dict.get('lotSizeUnit', '')
    
    if not lot_size or pd.isna(lot_size) or lot_size <= 0:
        return 0.0
    
    lot_size = float(lot_size)
    lot_unit = str(lot_unit).lower()
    
    if 'acre' in lot_unit:
        return lot_size
    elif 'sqft' in lot_unit or 'square' in lot_unit:
        return lot_size / 43560.0  # Convert sqft to acres
    elif 'sqm' in lot_unit:
        return lot_size / 4047.0   # Convert sqm to acres
    else:
        # Default to assuming sqft if unit is unclear
        return lot_size / 43560.0

def determine_property_context(lot_acres):
    """Determine if property is urban, suburban, or rural based on lot size."""
    if lot_acres >= 1.0:
        return 'rural'
    elif lot_acres >= 0.25:
        return 'suburban'
    else:
        return 'urban'

# --- Core Functions ---

def geocode_address(address, street=None, city=None, state=None, zipcode=None):
    """Geocodes a given address string to latitude and longitude using ArcGIS."""
    try:
        geolocator = ArcGIS()
        
        # Build full address string - use address parameter as primary street address
        address_parts = []
        if address:  # Use the address parameter (which contains the street address)
            address_parts.append(str(address))
        elif street:  # Fallback to street parameter if provided
            address_parts.append(str(street))
        if city:
            address_parts.append(str(city))
        if state:
            address_parts.append(str(state))
        if zipcode:
            address_parts.append(str(zipcode))
        
        full_addr = ', '.join([p for p in address_parts if p and p.strip()])
        
        print(f"DEBUG GEOCODING INPUT: Geocoding '{full_addr}'")
        
        location = geolocator.geocode(full_addr, timeout=10)
        
        if location:
            return location.latitude, location.longitude, 'ArcGIS'
        else:
            return None, None, f'Geocoding failed for: {full_addr}'
            
    except Exception as e:
        return None, None, f'Geocoding error: {str(e)}'

def load_data_from_database_wrapper():
    """Wrapper function to maintain compatibility with existing code."""
    return load_data_from_database()

def find_comps(subject_property, all_comps_df=None, return_all=False):
    """
    Find comparable properties using database queries.
    Uses optimized database search instead of filtering all data.
    """
    try:
        # Get subject property details
        lat = subject_property.get('latitude')
        lon = subject_property.get('longitude')
        if lat is None or lon is None:
            raise ValueError("Subject property is missing latitude/longitude")
        
        subject_coords = (lat, lon)
        subject_property_type = get_property_type(subject_property)
        today = pd.Timestamp.today().normalize()
        
        # Get properties using progressive radius search with proper comp selection radii
        comps_df = pd.DataFrame()
        search_radii = [0.5, 1.0, 2.0, 3.0, 5.0]  # Start with tight search like CSV version
        
        print(f"DEBUG: Finding comps for subject at {subject_coords}")
        
        for radius in search_radii:
            print(f"DEBUG: Searching radius {radius} miles...")
            temp_comps = find_comps_from_database(
                subject_property,
                radius_miles=radius,
                max_age_months=36
            )
            
            print(f"DEBUG: Found {len(temp_comps)} properties within {radius} miles")
            
            if not temp_comps.empty and len(temp_comps) >= 5:  # Need at least 5 for selection
                comps_df = temp_comps
                print(f"DEBUG: Using {len(comps_df)} properties from {radius} mile search")
                break  # Use first successful search with enough data
            elif not temp_comps.empty:
                # Keep accumulating if we don't have enough yet
                comps_df = temp_comps
        
        if comps_df.empty:
            raise ValueError(f"No properties found with propertyType '{subject_property_type}'")
        
        # Apply sophisticated scoring-based selection (like CSV version)
        selected_comps, backup_comps = apply_scoring_based_selection(
            subject_property, comps_df, today
        )
        
        if return_all:
            return selected_comps, backup_comps
        else:
            return selected_comps
            
    except Exception as e:
        raise ValueError(f"Failed to find comparables: {str(e)}")

def apply_scoring_based_selection(subject_property, comps_df, today):
    """Apply sophisticated scoring-based comp selection from CSV version."""
    
    subject_coords = (subject_property.get('latitude'), subject_property.get('longitude'))
    subject_beds = subject_property.get('bedrooms', 0)
    subject_baths = subject_property.get('bathrooms', 0)
    subject_sqft = subject_property.get('sqft', 0)
    subject_property_type = get_property_type(subject_property)
    
    # Calculate metrics for all comps
    comps_df = comps_df.copy()
    
    # Ensure we have distance and time calculations
    if 'distance_miles' not in comps_df.columns:
        comps_df['distance_miles'] = calculate_distance_miles(comps_df, subject_coords)
    if 'months_since_sale' not in comps_df.columns:
        comps_df['months_since_sale'] = calculate_months_since_sale(comps_df, today)
    
    # Extract lot size information
    subject_lot_acres = extract_lot_size_in_acres(subject_property)
    property_context = determine_property_context(subject_lot_acres)
    is_rural = property_context == 'rural'
    
    # Extract lot sizes for all comps
    comps_df['lot_acres'] = comps_df.apply(extract_lot_size_in_acres, axis=1)
    
    # Apply basic quality filters first (similar to CSV version)
    # Filter out very old properties (max 36 months)
    comps_df = comps_df[comps_df['months_since_sale'] <= 36]
    
    # Filter out very distant properties (max 5 miles for initial pool)
    comps_df = comps_df[comps_df['distance_miles'] <= 5.0]
    
    # Filter by bed/bath similarity (more lenient than tiered approach)
    if subject_property_type in RESIDENTIAL_PROPERTY_TYPES and subject_beds > 0:
        bed_tolerance = 3 if is_rural else 2  # Rural properties get more tolerance
        bath_tolerance = 3 if is_rural else 2
        
        comps_df = comps_df[
            (comps_df['bedrooms'].notna()) & (comps_df['bathrooms'].notna()) &
            (abs(comps_df['bedrooms'] - subject_beds) <= bed_tolerance) &
            (abs(comps_df['bathrooms'] - subject_baths) <= bath_tolerance)
        ]
    
    # Filter by square footage similarity (generous initial filter)
    if subject_sqft > 0:
        sqft_tolerance = 0.75  # ±75% initial filter
        sqft_min = subject_sqft * (1 - sqft_tolerance)
        sqft_max = subject_sqft * (1 + sqft_tolerance)
        comps_df = comps_df[(comps_df['sqft'] >= sqft_min) & (comps_df['sqft'] <= sqft_max)]
    
    if comps_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Apply sophisticated scoring system (from CSV version)
    def comp_score(row):
        # Age penalty - prefer recent sales
        age_penalty = row['months_since_sale'] * 5
        
        # Distance penalty - heavily favor proximity  
        distance_penalty = row['distance_miles'] * 100
        
        # If comp is very close (<1 mile), reduce age penalty significantly
        if row['distance_miles'] < 1.0:
            age_penalty *= 0.3  # 70% reduction for very close comps
        
        score = age_penalty + distance_penalty
        
        # Bed/bath/sqft similarity penalties
        if subject_property_type in RESIDENTIAL_PROPERTY_TYPES:
            score += abs(row['bedrooms'] - subject_beds) * 50
            score += abs(row['bathrooms'] - subject_baths) * 30
        
        # Square footage penalty (prefer similar size)
        if subject_sqft > 0 and row['sqft'] > 0:
            sqft_ratio = abs(row['sqft'] - subject_sqft) / subject_sqft
            score += sqft_ratio * 200  # Penalty for size differences
        
        # Lot size penalty for rural properties
        if is_rural and subject_lot_acres > 0.5 and row['lot_acres'] > 0:
            lot_ratio = max(row['lot_acres'] / subject_lot_acres, subject_lot_acres / row['lot_acres'])
            if lot_ratio > 2.0:
                score += 100  # Penalty for very different lot sizes
            elif lot_ratio > 1.5:
                score += 50   # Moderate penalty
        
        return score
    
    # Calculate scores for all comps
    comps_df['score'] = comps_df.apply(comp_score, axis=1)
    
    # Sort by score (lower is better) and distance for tie-breaking
    comps_df = comps_df.sort_values(['score', 'distance_miles'])
    
    # Debug: Show sample distances and scores
    if len(comps_df) > 0:
        print(f"DEBUG SCORING: Top 10 comps by score:")
        sample_comps = comps_df.head(10)[['streetAddress', 'distance_miles', 'months_since_sale', 'score']].to_dict('records')
        for i, comp in enumerate(sample_comps, 1):
            print(f"  {i}. {comp['streetAddress']}: {comp['distance_miles']:.3f} miles, {comp['months_since_sale']} months, score: {comp['score']:.1f}")
    
    # Select top 5 comps
    MIN_COMPS = 3
    MAX_COMPS = 5
    
    selected_comps = comps_df.head(MAX_COMPS).copy()
    print(f"DEBUG: Selected {len(selected_comps)} comps for final analysis")
    
    # Ensure we have minimum number of comps
    if len(selected_comps) < MIN_COMPS:
        # If we don't have enough, relax filters and try again
        backup_comps = comps_df.head(10).copy()  # Take more for backup
    else:
        # Use remaining comps as backup
        backup_comps = comps_df.iloc[MAX_COMPS:].head(10).copy()
    
    # Add tier labels based on distance/age for display
    def assign_tier_label(row):
        if row['distance_miles'] <= 0.5 and row['months_since_sale'] <= 12:
            return 'Strict'
        elif row['distance_miles'] <= 1.0 and row['months_since_sale'] <= 18:
            return 'Medium'
        elif row['distance_miles'] <= 2.0 and row['months_since_sale'] <= 24:
            return 'Loose'
        else:
            return 'Broad'
    
    selected_comps['comp_tier'] = selected_comps.apply(assign_tier_label, axis=1)
    if not backup_comps.empty:
        backup_comps['comp_tier'] = backup_comps.apply(assign_tier_label, axis=1)
    
    return selected_comps, backup_comps

def apply_tiered_filtering(subject_property, comps_df, today):
    """Apply tiered filtering to find best comparables."""
    
    subject_coords = (subject_property.get('latitude'), subject_property.get('longitude'))
    subject_beds = subject_property.get('bedrooms', 0)
    subject_baths = subject_property.get('bathrooms', 0)
    subject_sqft = subject_property.get('sqft', 0)
    subject_property_type = get_property_type(subject_property)
    
    # Determine if rural property
    subject_lot_acres = extract_lot_size_in_acres(subject_property)
    property_context = determine_property_context(subject_lot_acres)
    is_rural = property_context == 'rural'
    
    # Define tiers
    if is_rural:
        TIERS = [
            {'months': 12, 'miles': 0.5, 'bed_diff': 2, 'bath_diff': 2, 'sqft_tol': 0.20, 'name': 'Strict'},
            {'months': 18, 'miles': 1.0, 'bed_diff': 2, 'bath_diff': 2, 'sqft_tol': 0.25, 'name': 'Medium'},
            {'months': 24, 'miles': 2.0, 'bed_diff': 3, 'bath_diff': 2, 'sqft_tol': 0.49, 'name': 'Loose'},
            {'months': 36, 'miles': 3.0, 'bed_diff': 3, 'bath_diff': 3, 'sqft_tol': 0.75, 'name': 'Broad'},
        ]
    else:
        TIERS = [
            {'months': 12, 'miles': 0.5, 'bed_diff': 1, 'bath_diff': 1, 'sqft_tol': 0.20, 'name': 'Strict'},
            {'months': 18, 'miles': 1.0, 'bed_diff': 1, 'bath_diff': 1, 'sqft_tol': 0.25, 'name': 'Medium'},
            {'months': 24, 'miles': 2.0, 'bed_diff': 2, 'bath_diff': 2, 'sqft_tol': 0.49, 'name': 'Loose'},
            {'months': 36, 'miles': 3.0, 'bed_diff': 2, 'bath_diff': 2, 'sqft_tol': 0.75, 'name': 'Broad'},
        ]
    
    # Add distance and months since sale to comps
    comps_df['distance_miles'] = calculate_distance_miles(comps_df, subject_coords)
    comps_df['months_since_sale'] = calculate_months_since_sale(comps_df, today)
    
    selected_comps = pd.DataFrame()
    all_tier_comps = []
    
    # Try each tier
    for tier in TIERS:
        tier_comps = comps_df.copy()
        
        # Apply filters
        tier_comps = tier_comps[tier_comps['months_since_sale'] <= tier['months']]
        tier_comps = tier_comps[tier_comps['distance_miles'] <= tier['miles']]
        
        # Bed/bath filter for residential
        if subject_property_type in RESIDENTIAL_PROPERTY_TYPES:
            tier_comps = tier_comps[
                (abs(tier_comps['bedrooms'] - subject_beds) <= tier['bed_diff']) &
                (abs(tier_comps['bathrooms'] - subject_baths) <= tier['bath_diff'])
            ]
        
        # Square footage filter
        if subject_sqft > 0:
            sqft_min = subject_sqft * (1 - tier['sqft_tol'])
            sqft_max = subject_sqft * (1 + tier['sqft_tol'])
            tier_comps = tier_comps[
                (tier_comps['sqft'] >= sqft_min) & 
                (tier_comps['sqft'] <= sqft_max)
            ]
        
        if not tier_comps.empty:
            tier_comps['comp_tier'] = tier['name']
            all_tier_comps.append(tier_comps)
            
            if len(selected_comps) < 5:
                remaining_needed = 5 - len(selected_comps)
                selected_comps = pd.concat([selected_comps, tier_comps.head(remaining_needed)])
    
    # Create backup comps from remaining properties
    backup_comps = pd.DataFrame()
    if all_tier_comps:
        all_available = pd.concat(all_tier_comps)
        used_zpids = set(selected_comps['zpid']) if 'zpid' in selected_comps.columns else set()
        backup_comps = all_available[~all_available['zpid'].isin(used_zpids)]
    
    return selected_comps, backup_comps

def determine_comp_tier(comp_row, subject_property, today=None):
    """Determine which tier a comp belongs to."""
    if today is None:
        today = pd.Timestamp.today().normalize()
    
    subject_coords = (subject_property.get('latitude'), subject_property.get('longitude'))
    subject_beds = subject_property.get('bedrooms', 0)
    subject_baths = subject_property.get('bathrooms', 0)
    subject_sqft = subject_property.get('sqft', 0)
    subject_property_type = get_property_type(subject_property)
    
    # Calculate metrics for this comp
    comp_date = pd.to_datetime(comp_row.get('dateSold'), errors='coerce')
    if pd.isnull(comp_date):
        return None
        
    months_since_sale = (today.year - comp_date.year) * 12 + (today.month - comp_date.month)
    distance_miles = geodesic((comp_row.get('latitude'), comp_row.get('longitude')), subject_coords).miles
    
    # Determine property context
    subject_lot_acres = extract_lot_size_in_acres(subject_property)
    property_context = determine_property_context(subject_lot_acres)
    is_rural = property_context == 'rural'
    
    # Define tiers (same as in apply_tiered_filtering)
    if is_rural:
        TIERS = [
            {'months': 12, 'miles': 0.5, 'bed_diff': 2, 'bath_diff': 2, 'sqft_tol': 0.20, 'name': 'Strict'},
            {'months': 18, 'miles': 1.0, 'bed_diff': 2, 'bath_diff': 2, 'sqft_tol': 0.25, 'name': 'Medium'},
            {'months': 24, 'miles': 2.0, 'bed_diff': 3, 'bath_diff': 2, 'sqft_tol': 0.49, 'name': 'Loose'},
            {'months': 36, 'miles': 3.0, 'bed_diff': 3, 'bath_diff': 3, 'sqft_tol': 0.75, 'name': 'Broad'},
        ]
    else:
        TIERS = [
            {'months': 12, 'miles': 0.5, 'bed_diff': 1, 'bath_diff': 1, 'sqft_tol': 0.20, 'name': 'Strict'},
            {'months': 18, 'miles': 1.0, 'bed_diff': 1, 'bath_diff': 1, 'sqft_tol': 0.25, 'name': 'Medium'},
            {'months': 24, 'miles': 2.0, 'bed_diff': 2, 'bath_diff': 2, 'sqft_tol': 0.49, 'name': 'Loose'},
            {'months': 36, 'miles': 3.0, 'bed_diff': 2, 'bath_diff': 2, 'sqft_tol': 0.75, 'name': 'Broad'},
        ]
    
    # Check each tier
    for tier in TIERS:
        if months_since_sale > tier['months'] or distance_miles > tier['miles']:
            continue
            
        if subject_property_type in RESIDENTIAL_PROPERTY_TYPES:
            comp_beds = comp_row.get('bedrooms', 0)
            comp_baths = comp_row.get('bathrooms', 0)
            if (abs(comp_beds - subject_beds) > tier['bed_diff'] or 
                abs(comp_baths - subject_baths) > tier['bath_diff']):
                continue
        
        comp_sqft = comp_row.get('sqft', 0)
        if subject_sqft > 0 and comp_sqft > 0:
            sqft_min = subject_sqft * (1 - tier['sqft_tol'])
            sqft_max = subject_sqft * (1 + tier['sqft_tol'])
            if comp_sqft < sqft_min or comp_sqft > sqft_max:
                continue
        
        return tier['name']
    
    return 'beyond_broad'

# --- Simplified placeholder functions ---
# These would need full implementation from the original file

def fetch_subject_property_from_api(subject_property, api_key):
    """
    Fetch subject property info from the RapidAPI endpoint using the address fields in subject_property.
    Returns a dict matching the app's property/comps structure, or raises an error if the call fails.
    """
    # Build the address string
    address = f"{subject_property.get('streetAddress', '')}, {subject_property.get('city', '')}, {subject_property.get('state', '')}, {subject_property.get('zipcode', '')}"
    encoded_address = urllib.parse.quote(address)
    url = f"https://zillow-com4.p.rapidapi.com/properties/search-address?address={encoded_address}"
    headers = {
        "x-rapidapi-host": "zillow-com4.p.rapidapi.com",
        "x-rapidapi-key": api_key,
    }
    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"RapidAPI call failed: {response.status_code} {response.text}")
    api_response = response.json()
    if not api_response.get("status") or "data" not in api_response:
        raise RuntimeError(f"RapidAPI call did not return valid data: {api_response}")
    data = api_response["data"]

    # Extraction logic (from extract_property_info.py)
    def get_quick_fact(quick_facts, element_type):
        for fact in quick_facts:
            if fact.get("elementType") == element_type:
                return fact.get("value", {}).get("fullValue")
        return None

    property_type_map = {
        'SINGLE_FAMILY': 'singleFamily',
        'MULTI_FAMILY': 'multiFamily',
        'LAND': 'land',
        # Add more as needed
    }

    location = data.get("formattedChip", {}).get("location", [])
    street = location[0]["fullValue"] if len(location) > 0 else ""
    city_state_zip = location[1]["fullValue"] if len(location) > 1 else ""
    city = city_state_zip.split(",")[0].strip() if city_state_zip else ""
    state = city_state_zip.split(",")[1].split()[0] if city_state_zip and "," in city_state_zip else ""
    zipcode = city_state_zip.split()[-1] if city_state_zip else ""
    zipcode = str(zipcode) if zipcode is not None else ""

    quick_facts = data.get("formattedChip", {}).get("quickFacts", [])
    bedrooms = get_quick_fact(quick_facts, "beds")
    bathrooms = get_quick_fact(quick_facts, "baths")
    sqft = get_quick_fact(quick_facts, "livingArea")

    if bedrooms is not None:
        try:
            bedrooms = int(bedrooms)
        except (ValueError, TypeError):
            bedrooms = None

    if bathrooms is not None:
        try:
            bathrooms = float(bathrooms)
        except (ValueError, TypeError):
            bathrooms = None

    if sqft is not None:
        try:
            sqft = int(sqft.replace(",", ""))
        except (ValueError, TypeError, AttributeError):
            sqft = None

    # Get lot size
    lot_size = get_quick_fact(quick_facts, "lotSize")
    lot_size_unit = None
    if lot_size:
        lot_size_str = str(lot_size)
        if "acre" in lot_size_str.lower():
            lot_size_unit = "acres"
            try:
                lot_size = float(lot_size_str.split()[0])
            except:
                lot_size = None
        elif "sqft" in lot_size_str.lower():
            lot_size_unit = "sqft"
            try:
                lot_size = int(lot_size_str.replace(",", "").split()[0])
            except:
                lot_size = None
        else:
            lot_size = None

    # Get property type
    property_type_raw = data.get("propertyTypeDimension")
    property_type = property_type_map.get(property_type_raw, 'singleFamily')

    # Get year built
    year_built = get_quick_fact(quick_facts, "yearBuilt")
    if year_built:
        try:
            year_built = int(year_built)
        except:
            year_built = None

    # Extract image URLs - prioritize high res, fallback to regular photos
    image_urls = []
    try:
        # First try high resolution photos
        high_res_photos = data.get("photoUrlsHighRes", [])
        if isinstance(high_res_photos, list) and high_res_photos:
            image_urls = [photo.get("url") for photo in high_res_photos if isinstance(photo, dict) and photo.get("url")]
        
        # If no high res photos, try regular photos
        if not image_urls:
            regular_photos = data.get("photoUrls", [])
            if isinstance(regular_photos, list) and regular_photos:
                image_urls = [photo.get("url") for photo in regular_photos if isinstance(photo, dict) and photo.get("url")]
        
        # Filter for valid HTTP URLs with common image extensions
        image_urls = [url for url in image_urls if isinstance(url, str) and url.startswith("http") and any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp'])]
        
    except Exception as e:
        print(f"DEBUG: Error extracting image URLs: {e}")
        pass

    return {
        'streetAddress': street,
        'city': city,
        'state': state,
        'zipcode': zipcode,
        'propertyType': property_type,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft': sqft,
        'lotSize': lot_size,
        'lotSizeUnit': lot_size_unit,
        'yearBuilt': year_built,
        'image_urls': image_urls
    }

def run_full_analysis(subject_property, all_comps_df, num_runs, progress_callback=None):
    """Simplified analysis function without OpenAI."""
    try:
        # Use coordinates from RapidAPI if available, otherwise geocode
        if 'latitude' in subject_property and 'longitude' in subject_property and subject_property['latitude'] and subject_property['longitude']:
            # Use RapidAPI coordinates
            lat = subject_property['latitude']
            lon = subject_property['longitude']
            if progress_callback:
                progress_callback("Using coordinates from RapidAPI...")
        else:
            # Fallback to geocoding
            if progress_callback:
                progress_callback("Geocoding subject property...")
            
            lat, lon, source = geocode_address(
                address=subject_property['streetAddress'],
                city=subject_property.get('city'),
                state=subject_property.get('state'),
                zipcode=subject_property.get('zipcode')
            )
            
            if lat is None or lon is None:
                raise ValueError(f"Failed to geocode address: {source}")
            
            print(f"DEBUG GEOCODING: Subject property geocoded to: lat={lat}, lon={lon} via {source}")
            subject_property['latitude'] = lat
            subject_property['longitude'] = lon
        
        # Find comparables
        if progress_callback:
            progress_callback("Finding comparable properties...")
        
        selected_comps, backup_comps = find_comps(subject_property, return_all=True)
        
        if selected_comps.empty:
            raise ValueError("No comparable properties found")
        
        # Simplified valuation without AI
        if progress_callback:
            progress_callback("Calculating ARV...")
        
        # Filter out invalid prices and sqft
        valid_comps = selected_comps[
            (selected_comps['price'] > 0) & 
            (selected_comps['sqft'] > 0) & 
            pd.notnull(selected_comps['price']) & 
            pd.notnull(selected_comps['sqft'])
        ]
        
        if valid_comps.empty:
            raise ValueError("No valid comparable properties with price and sqft data")
        
        # Pre-load image descriptions for ALL comps (selected + backup) for comp swapping
        if progress_callback:
            progress_callback("Pre-loading image descriptions for all comparables...")
        
        all_comps_for_images = pd.concat([selected_comps, backup_comps]) if not backup_comps.empty else selected_comps
        for idx, row in all_comps_for_images.iterrows():
            if 'image_description' not in all_comps_for_images.columns or pd.isnull(row.get('image_description', None)) or not row.get('image_description', None):
                desc = get_image_description(row.get('image_urls', []), row)
                all_comps_for_images.at[idx, 'image_description'] = desc
        
        # Update backup_comps with image descriptions
        if not backup_comps.empty:
            backup_comps = all_comps_for_images[all_comps_for_images.index.isin(backup_comps.index)].copy()
        
        # Get subject property image description if needed
        if 'image_description' not in subject_property or not subject_property.get('image_description'):
            desc = get_image_description(subject_property.get('image_urls', []), subject_property)
            subject_property['image_description'] = desc
        
        # Run multiple analysis iterations
        all_arvs = []
        all_run_details = []
        all_comps_with_runs = []
        
        subject_sqft = subject_property.get('sqft', 1000)
        if subject_sqft <= 0:
            subject_sqft = 1000  # Default if missing
        
        for run_num in range(num_runs):
            if progress_callback:
                progress_callback(f"Running analysis iteration {run_num + 1} of {num_runs}...")
            
            # Apply OpenAI analysis to get fresh adjustments and weights for this run
            adjustment_result = get_structured_adjustments(subject_property, valid_comps)
            adjustment_map = adjustment_result['adjustments']
            arv_summary = adjustment_result.get('arv_summary', 'ARV calculated using weighted average of adjusted comparable prices.')
            
            # Apply AI adjustments to comp prices (fresh for each run)
            def safe_get(addr, field, default):
                return adjustment_map.get(addr, {}).get(field, default)
            
            run_comps = valid_comps.copy()
            run_comps['ai_adjustment'] = run_comps['streetAddress'].apply(lambda addr: safe_get(addr, 'adjustment', 0))
            run_comps['ai_weight'] = run_comps['streetAddress'].apply(lambda addr: safe_get(addr, 'weight', 1.0 / len(run_comps)))
            run_comps['ai_explanation'] = run_comps['streetAddress'].apply(lambda addr: safe_get(addr, 'explanation', 'No explanation available'))
            run_comps['run'] = run_num + 1  # Add run number
            
            # Apply adjustments to prices
            run_comps['adjusted_price'] = run_comps['price'] + run_comps['ai_adjustment']
            run_comps['price_per_sqft'] = run_comps['adjusted_price'] / run_comps['sqft']
            
            # Calculate weighted average using AI weights
            if run_comps['ai_weight'].sum() > 0:
                weighted_avg_price_per_sqft = (
                    (run_comps['price_per_sqft'] * run_comps['ai_weight']).sum() / 
                    run_comps['ai_weight'].sum()
                )
            else:
                # Fallback to simple average if AI weighting fails
                weighted_avg_price_per_sqft = run_comps['price_per_sqft'].mean()
            
            # Final ARV calculation using AI-weighted average
            estimated_arv = weighted_avg_price_per_sqft * subject_sqft
            all_arvs.append(int(estimated_arv))
            
            # Store run details for report generation
            run_details = {
                'run_num': run_num + 1,
                'arv': estimated_arv,
                'price_per_sqft': weighted_avg_price_per_sqft,
                'arv_summary': arv_summary,
                'comps': run_comps.copy()
            }
            all_run_details.append(run_details)
            all_comps_with_runs.append(run_comps)
        
        # Calculate statistics from multiple runs
        mean_arv = np.mean(all_arvs)
        min_arv = min(all_arvs)
        max_arv = max(all_arvs)
        
        # Calculate IQR
        if len(all_arvs) >= 3:
            q1 = np.percentile(all_arvs, 25)
            q3 = np.percentile(all_arvs, 75)
        else:
            # For 1-2 runs, use ±5% of mean
            q1 = mean_arv * 0.95
            q3 = mean_arv * 1.05
        
        # Format subject property details for report
        full_address = f"{subject_property.get('streetAddress', 'Unknown')}, {subject_property.get('city', 'Unknown')}, {subject_property.get('state', 'Unknown')}, {subject_property.get('zipcode', 'Unknown')}"
        lot_size_display = subject_property.get('lotSize', 'N/A')
        if subject_property.get('lotSizeUnit'):
            lot_size_display += f" {subject_property.get('lotSizeUnit')}"
        
        # Generate comprehensive multi-run summary report
        run_label = f"{num_runs}-Run Average" if num_runs > 1 else "1-Run Average"
        summary_report = f"""--- Underwriting Summary Report ({run_label}) ---
Subject Property: {full_address}
  - {subject_property.get('bedrooms', 'N/A')} bed, {subject_property.get('bathrooms', 'N/A')} bath, {subject_sqft} sqft

"""
        
        # Add individual run ARVs
        for i, arv in enumerate(all_arvs, 1):
            summary_report += f"Run {i} ARV: ${arv:,.2f}\n"
        
        summary_report += f"\n--- FINAL AVERAGED AFTER-REPAIR VALUE (ARV): ${mean_arv:,.2f} ---"
        
        if num_runs > 1:
            summary_report += f"""
Range: ${min_arv:,.2f} - ${max_arv:,.2f}
IQR: ${q1:,.2f} - ${q3:,.2f}"""
        
        summary_report += "\n\n--- Detailed Run-by-Run Analysis ---\n"
        
        # Add detailed analysis for each run
        for run_detail in all_run_details:
            run_num = run_detail['run_num']
            run_arv = run_detail['arv']
            run_price_per_sqft = run_detail['price_per_sqft']
            run_arv_summary = run_detail['arv_summary']
            
            summary_report += f"""
--- Individual Run {run_num} ARV: ${run_arv:,.2f} ---
Subject Property: {full_address}
  Beds: {subject_property.get('bedrooms', 'N/A')}, Baths: {subject_property.get('bathrooms', 'N/A')}, Sqft: {subject_sqft:,}, Lot Size: {lot_size_display} 
  Calculated ARV: ${run_arv:,.2f}, Price/Sqft: ${run_price_per_sqft:.2f}
  Image Analysis: {subject_property.get('image_description', 'No image analysis available for subject property.')}

ARV Calculation Summary: {run_arv_summary}

Valuation Method: AI-Enhanced Weighted Analysis
AI-Weighted Average Price/Sqft: ${run_price_per_sqft:.2f}
Final ARV: ${run_arv:,.2f}
"""
        
        # Add comparable summary using data from the first run
        first_run_comps = all_run_details[0]['comps']
        summary_report += generate_comparable_summary_table(first_run_comps)
        
        # Add detailed AI adjustment explanations for each comp
        summary_report += "\n\n--- AI Adjustment Details ---\n"
        for _, comp in first_run_comps.iterrows():
            addr = comp.get('streetAddress', 'N/A')
            ai_adj = comp.get('ai_adjustment', 0)
            ai_weight = comp.get('ai_weight', 0)
            ai_explanation = comp.get('ai_explanation', 'No explanation available')
            
            adj_sign = "+" if ai_adj >= 0 else ""
            summary_report += f"\n{addr}:\n"
            summary_report += f"  • AI Adjustment: {adj_sign}${ai_adj:,}\n"
            summary_report += f"  • AI Weight: {ai_weight:.3f}\n"
            summary_report += f"  • Explanation: {ai_explanation}\n"
        
        # Combine all runs' comps data for final_comps_df
        combined_comps_df = pd.concat(all_comps_with_runs, ignore_index=True) if all_comps_with_runs else valid_comps
        
        # Return multi-run result
        return {
            'best_arv': int(mean_arv),
            'arv_range': (int(min_arv), int(max_arv)),
            'arv_iqr': (int(q1), int(q3)),
            'confidence': 3,
            'all_arvs': all_arvs,
            'summary_report': summary_report,
            'final_comps_df': combined_comps_df,
            'backup_comps': backup_comps,
            'subject_images': subject_property.get('image_urls', [])
        }
        
    except Exception as e:
        raise ValueError(f"Analysis failed: {str(e)}")

def get_image_description(image_urls, property_info):
    """Sends property images to GPT-4o and returns a description."""
    if OPENAI_API_KEY == "" or not OPENAI_API_KEY:
        return "Skipping image analysis because OpenAI API key is not set."

    # Parse stringified list if needed
    if isinstance(image_urls, str):
        try:
            image_urls = ast.literal_eval(image_urls)
        except Exception:
            image_urls = []
    if not isinstance(image_urls, list):
        image_urls = []

    # Filter out invalid/empty image URLs
    valid_image_urls = [u for u in image_urls if isinstance(u, str) and u.strip().lower().startswith("http") and ".jpg" in u.lower()]

    if not valid_image_urls:
        return "No valid images provided for analysis."

    prompt_text = (
        "You are a professional real estate underwriter with 15+ years of experience in property valuation. "
        "Analyze these property photos from an underwriting perspective, focusing on factors that impact market value. "
        "Provide a concise professional assessment in 2-3 sentences covering: "
        "\n• CONDITION: Overall maintenance, visible defects, and repair needs that affect value "
        "\n• QUALITY: Architectural style, materials, craftsmanship, and finish level relative to market standards "
        "\n• MARKETABILITY: Curb appeal, design features, and elements that influence buyer desirability "
        "\nFocus only on permanent property features that impact valuation. "
        "Do NOT comment on furnishings, staging, or personal items. "
        "Be objective and use professional real estate terminology.\n\n"
        f"Property Details: {property_info}"
    )
    # Build content_payload as a list of message parts for OpenAI vision API
    content_payload = []
    content_payload.append({"type": "text", "text": prompt_text})
    for url in valid_image_urls[:10]:
        content_payload.append({"type": "image_url", "image_url": {"url": url}})
    try:
        if not openai_client:
            return "OpenAI API key not configured"
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content_payload}],
            max_tokens=250,
            temperature=0.4
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            content = content.strip()
        else:
            content = ""
        return content
    except Exception as e:
        return f"Error calling OpenAI API for image description: {e}"

def generate_comparable_summary_table(comps_df):
    """
    Generate the detailed comparable summary table with consistent formatting.
    Used by both initial analysis and recompute ARV to maintain consistent format.
    """
    table_str = "\n--- Comparable Summary ---\n"
    table_str += "      streetAddress zipcode    city state  bedrooms  bathrooms   sqft  lotSize  distance_miles     price   dateSold   comp_tier                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             image_description  ai_weight  ai_adjustment                                                                                          ai_explanation"
    
    for _, comp in comps_df.iterrows():
        # Format the comp data similar to CSV version
        street_address = comp.get('streetAddress', 'N/A')
        zipcode = comp.get('zipcode', 'N/A')
        city = comp.get('city', 'N/A')
        state = comp.get('state', 'N/A')
        bedrooms = comp.get('bedrooms', 'N/A')
        bathrooms = comp.get('bathrooms', 'N/A')
        sqft = comp.get('sqft', 'N/A')
        lot_size = comp.get('lotSize', 'N/A')
        if lot_size == 'N/A' or pd.isna(lot_size):
            lot_size = 'N/A'
        distance_miles = comp.get('distance_miles', 'N/A')
        if distance_miles != 'N/A' and not pd.isna(distance_miles):
            distance_miles = f"{distance_miles:.6f}"
        price = comp.get('price', 'N/A')
        if price != 'N/A' and not pd.isna(price):
            price = f"{price:.1f}"
        date_sold = comp.get('dateSold', 'N/A')
        comp_tier = comp.get('comp_tier', 'N/A')
        
        # AI analysis data
        image_description = comp.get('image_description', 'Image analysis not available')
        ai_weight = comp.get('ai_weight', 'N/A')
        if ai_weight != 'N/A':
            ai_weight = f"{ai_weight:.3f}"
        ai_adjustment = comp.get('ai_adjustment', 0)
        if ai_adjustment != 0:
            ai_adjustment = f"{ai_adjustment:+d}"
        else:
            ai_adjustment = "0"
        ai_explanation = comp.get('ai_explanation', 'No AI explanation available')
        
        # Build the comp line
        comp_line = f"""
{street_address}   {zipcode} {city}    {state}       {bedrooms}        {bathrooms} {sqft} {lot_size}        {distance_miles} {price} {date_sold} {comp_tier}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             {image_description}   {ai_weight}         {ai_adjustment}                                                                                          {ai_explanation}"""
        
        table_str += comp_line
    
    return table_str

def get_structured_adjustments(subject_property, comps_df, subject_property_source='manual'):
    """Formulate a summary of comps and subject for ARV calculation. Includes dateSold and image info."""
    def format_property_info(row_series):
        info = f"Address: {row_series.get('streetAddress', 'N/A')}, Beds: {row_series.get('bedrooms', 'N/A')}, Baths: {row_series.get('bathrooms', 'N/A')}, Sqft: {row_series.get('sqft', 'N/A')}, Price: {row_series.get('price', 'N/A')}"
        if 'image_description' in row_series and row_series['image_description']:
            info += f", Image Description: {row_series['image_description']}"
        return info
    
    # For each comp, get image description if not present
    for idx, row in comps_df.iterrows():
        if 'image_description' not in comps_df.columns or pd.isnull(row.get('image_description', None)) or not row.get('image_description', None):
            desc = get_image_description(row['image_urls'], row)
            comps_df.at[idx, 'image_description'] = desc
    
    # For subject property, always get image description if image_urls present and not already described
    if 'image_description' not in subject_property or not subject_property.get('image_description'):
        desc = get_image_description(subject_property.get('image_urls', []), subject_property)
        subject_property['image_description'] = desc
    
    subject_info_str = format_property_info(pd.Series(subject_property))
    comp_info_list = [format_property_info(row) for _, row in comps_df.iterrows()]

    # Compose prompt for OpenAI
    system_prompt = (
        "You are a senior real estate underwriter with MAI certification and 20+ years of commercial and residential valuation experience. "
        "You specialize in comparative market analysis and property adjustments. "
        "Provide professional underwriting analysis in strict JSON format only. No explanations or markdown."
    )
    if subject_property_source == "api":
        subject_context = "The subject property information below was retrieved from a trusted real estate API and should be considered the ground truth for comparison."
    else:
        subject_context = "The subject property information below was entered manually by the user."
    
    user_prompt = (
        f"UNDERWRITING ASSIGNMENT: Comparative Market Analysis\n"
        f"{subject_context}\n\n"
        "INSTRUCTIONS:\n"
        "Analyze each comparable property relative to the subject and provide:\n"
        "1. ADJUSTMENT (integer, in dollars): Net adjustment to comp's sale price\n"
        "2. EXPLANATION (string): Professional justification for adjustment\n"
        "3. WEIGHT (float, 0-1): Reliability/relevance weight for this comp\n\n"
        "ADJUSTMENT METHODOLOGY:\n"
        "• NEGATIVE adjustment: Comp is superior to subject (reduce comp's value to match subject)\n"
        "• POSITIVE adjustment: Comp is inferior to subject (increase comp's value to match subject)\n"
        "• DO NOT adjust for square footage (handled by regression model)\n"
        "• Focus on: bed/bath count, condition, quality, age/updates, lot characteristics\n\n"
        "CONDITION & QUALITY ANALYSIS:\n"
        "• When Image Descriptions are available, use them to assess condition and finish quality differences\n"
        "• Consider: exterior maintenance, architectural details, visible updates, overall property presentation\n"
        "• Weight image-based insights at 20-30% of total adjustment (not the primary factor)\n"
        "• Primary factors remain: bed/bath count, age, lot size, location proximity\n"
        "• When 'No valid images provided', rely on quantitative factors only\n\n"
        "WEIGHTING CRITERIA:\n"
        "• Higher weight: Recent sales, closer distance, similar bed/bath/age, better condition data\n"
        "• Lower weight: Older sales, distant location, significant size differences, missing information\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "Return JSON with keys: 'adjustments' (comp address -> {adjustment, explanation, weight}) and 'arv_summary' (methodology explanation)\n\n"
        "PROPERTY DATA:\n"
        f"Subject Property:\n{subject_info_str}\n\n"
        f"Comparable Properties:\n" + "\n".join([f"{i+1}. {comp}" for i, comp in enumerate(comp_info_list)])
    )

    if OPENAI_API_KEY == "" or not OPENAI_API_KEY:
        # Fallback: return neutral values
        adjustment_map = {}
        for _, comp in comps_df.iterrows():
            addr = comp.get('streetAddress', comp.get('street_address', 'N/A'))
            adjustment_map[addr] = {'adjustment': 0, 'explanation': '[OpenAI disabled]', 'weight': 1.0 / len(comps_df) if len(comps_df) else 1.0}
        return {'adjustments': adjustment_map, 'arv_summary': '[OpenAI disabled - ARV calculated using weighted average of adjusted comparable prices with non-linear regression for square footage adjustments]'}

    try:
        if not openai_client:
            return {'adjustments': {}, 'arv_summary': 'OpenAI API key not configured - using basic valuation'}
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.2
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            content = content.strip()
        else:
            content = ""
        
        try:
            result = json.loads(content)
            # Handle both old format (just adjustments) and new format (adjustments + arv_summary)
            if 'adjustments' in result and 'arv_summary' in result:
                return result
            else:
                # Old format - wrap in new structure
                return {'adjustments': result, 'arv_summary': 'ARV calculated using weighted average of adjusted comparable prices with non-linear regression for square footage adjustments.'}
        except Exception:
            # Fallback if JSON parsing fails
            adjustment_map = {}
            for _, comp in comps_df.iterrows():
                addr = comp.get('streetAddress', comp.get('street_address', 'N/A'))
                adjustment_map[addr] = {'adjustment': 0, 'explanation': 'JSON parse error', 'weight': 1.0 / len(comps_df)}
            return {'adjustments': adjustment_map, 'arv_summary': 'ARV calculated using weighted average (OpenAI response could not be parsed)'}
    
    except Exception as e:
        # Error fallback
        adjustment_map = {}
        for _, comp in comps_df.iterrows():
            addr = comp.get('streetAddress', comp.get('street_address', 'N/A'))
            adjustment_map[addr] = {'adjustment': 0, 'explanation': f'OpenAI error: {str(e)[:50]}', 'weight': 1.0 / len(comps_df)}
        return {'adjustments': adjustment_map, 'arv_summary': f'ARV calculated using weighted average (OpenAI error: {str(e)[:50]})'}

def perform_underwriting(subject_property, comps_df):
    """Simplified underwriting calculation."""
    if comps_df.empty:
        return 0, comps_df
    
    # Simple average price per sqft
    comps_df['price_per_sqft'] = comps_df['price'] / comps_df['sqft']
    avg_price_per_sqft = comps_df['price_per_sqft'].mean()
    
    subject_sqft = subject_property.get('sqft', 1000)
    arv = avg_price_per_sqft * subject_sqft
    
    return int(arv), comps_df

def generate_single_run_report(subject_property, arv, comps_df, arv_summary):
    """Generate simple analysis report."""
    return f"""
Property Analysis Report
========================

Subject Property: {subject_property.get('streetAddress', 'Unknown Address')}
Estimated ARV: ${arv:,}

Comparable Properties Used: {len(comps_df)}

{arv_summary}

Note: This is a simplified database version. Full AI analysis features coming soon.
"""

# Compatibility function
def load_and_prepare_data(csv_path=None, debug=False):
    """Compatibility wrapper that loads from database instead of CSV."""
    if debug:
        print("[DEBUG] Loading data from database instead of CSV")
    return load_data_from_database_wrapper()