"""
Database client for Supabase integration
Handles all database connections and queries for the AI Underwriting Agent
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from supabase import create_client, Client
from typing import Optional, Dict, Any, List
import os

# Handle Streamlit import for secrets
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class DatabaseClient:
    """Handles all database operations for the underwriting agent"""
    
    def __init__(self, url=None, key=None):
        """Initialize Supabase client using provided credentials or Streamlit secrets"""
        if url and key:
            # Use provided credentials (for migration script)
            self.url = url
            self.key = key
        elif STREAMLIT_AVAILABLE:
            # Use Streamlit secrets
            self.url = st.secrets["supabase"]["url"]
            self.key = st.secrets["supabase"]["anon_key"]
        else:
            # Use environment variables as fallback
            self.url = os.getenv("SUPABASE_URL")
            self.key = os.getenv("SUPABASE_ANON_KEY")
            if not self.url or not self.key:
                raise ValueError("No Supabase credentials found. Provide url/key parameters or set environment variables.")
        
        self.client: Client = create_client(self.url, self.key)
    
    def get_all_properties(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load all properties from the database
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            pandas.DataFrame: Properties data with standardized column names
        """
        try:
            query = self.client.table('properties').select('*')
            
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            
            if not result.data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(result.data)
            
            # Standardize column names for compatibility with existing code
            df = self._standardize_columns(df)
            
            # Convert data types
            df = self._convert_data_types(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Database query failed: {str(e)}")
    
    def get_properties_by_location(
        self, 
        latitude: float, 
        longitude: float, 
        radius_miles: float = 10.0,
        max_age_months: int = 36,
        property_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Find properties within a geographic radius and time window
        
        Args:
            latitude: Subject property latitude
            longitude: Subject property longitude
            radius_miles: Search radius in miles
            max_age_months: Maximum age of sold properties in months
            property_type: Filter by property type
            
        Returns:
            pandas.DataFrame: Filtered properties
        """
        try:
            # Calculate date cutoff
            cutoff_date = datetime.now() - timedelta(days=max_age_months * 30)
            
            # Base query
            query = self.client.table('properties').select('*')
            
            # Filter by sold date
            query = query.gte('last_sold_date', cutoff_date.strftime('%Y-%m-%d'))
            
            # Filter by property type if specified
            if property_type:
                query = query.eq('property_type', property_type)
            
            result = query.execute()
            
            if not result.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(result.data)
            df = self._standardize_columns(df)
            df = self._convert_data_types(df)
            
            # Calculate distances and filter by radius
            if not df.empty:
                from geopy.distance import geodesic
                subject_coords = (latitude, longitude)
                
                df['distance_miles'] = df.apply(
                    lambda row: geodesic(
                        (row['latitude'], row['longitude']), 
                        subject_coords
                    ).miles, 
                    axis=1
                )
                
                # Filter by radius
                df = df[df['distance_miles'] <= radius_miles]
            
            return df
            
        except Exception as e:
            raise Exception(f"Location-based query failed: {str(e)}")
    
    def get_property_images(self, property_ids: List[int]) -> pd.DataFrame:
        """
        Get images for specific properties
        
        Args:
            property_ids: List of property IDs
            
        Returns:
            pandas.DataFrame: Property images data
        """
        try:
            if not property_ids:
                return pd.DataFrame()
                
            query = self.client.table('property_images').select('*')
            query = query.in_('property_id', property_ids)
            
            result = query.execute()
            
            if not result.data:
                return pd.DataFrame()
            
            return pd.DataFrame(result.data)
            
        except Exception as e:
            raise Exception(f"Image query failed: {str(e)}")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize database column names to match expected format
        
        Args:
            df: Raw dataframe from database
            
        Returns:
            pandas.DataFrame: DataFrame with standardized column names
        """
        # Map database column names to expected format
        column_mapping = {
            'id': 'id',
            'zpid': 'zpid',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'street_address': 'streetAddress',
            'city': 'city',
            'state': 'state',
            'zipcode': 'zipcode',
            'bedrooms': 'bedrooms',
            'bathrooms': 'bathrooms',
            'living_area': 'sqft',  # Map living_area to sqft for compatibility
            'year_built': 'yearBuilt',
            'lot_size': 'lotSize',
            'lot_size_unit': 'lotSizeUnit',
            'property_type': 'propertyType',
            'price': 'price',
            'last_sold_date': 'dateSold',
            'listing_status': 'listingStatus',
            'tax_assessed_value': 'taxAssessedValue',
            'tax_assessment_year': 'taxAssessmentYear',
            'hdp_url': 'hdpUrl',
            'created_at': 'created_at',
            'updated_at': 'updated_at'
        }
        
        # Rename columns that exist
        existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mappings)
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types for proper analysis
        
        Args:
            df: DataFrame with raw data types
            
        Returns:
            pandas.DataFrame: DataFrame with converted data types
        """
        if df.empty:
            return df
        
        # Convert numeric columns
        numeric_columns = ['latitude', 'longitude', 'bedrooms', 'bathrooms', 
                          'sqft', 'yearBuilt', 'lotSize', 'price', 'taxAssessedValue']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date columns
        date_columns = ['dateSold']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df


# Global instance
_db_client = None

def get_database_client() -> DatabaseClient:
    """Get or create database client singleton"""
    global _db_client
    if _db_client is None:
        _db_client = DatabaseClient()
    return _db_client


def load_data_from_database(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Main function to load property data from database
    Replaces the CSV-based load_and_prepare_data function
    
    Args:
        limit: Maximum number of records to load
        
    Returns:
        pandas.DataFrame: Standardized property data
    """
    try:
        client = get_database_client()
        df = client.get_all_properties(limit=limit)
        
        if df.empty:
            raise ValueError("No property data found in database")
        
        return df
        
    except Exception as e:
        raise Exception(f"Failed to load data from database: {str(e)}")


def find_comps_from_database(
    subject_property: Dict[str, Any],
    radius_miles: float = 5.0,
    max_age_months: int = 24
) -> pd.DataFrame:
    """
    Find comparable properties from database using geographic search
    Optimized database query instead of loading all data
    
    Args:
        subject_property: Subject property details
        radius_miles: Search radius in miles
        max_age_months: Maximum age of comparables
        
    Returns:
        pandas.DataFrame: Comparable properties
    """
    try:
        lat = subject_property.get('latitude')
        lon = subject_property.get('longitude')
        
        if lat is None or lon is None:
            raise ValueError("Subject property missing latitude/longitude")
        
        property_type = subject_property.get('propertyType', 'singleFamily')
        
        client = get_database_client()
        comps_df = client.get_properties_by_location(
            latitude=lat,
            longitude=lon,
            radius_miles=radius_miles,
            max_age_months=max_age_months,
            property_type=property_type
        )
        
        return comps_df
        
    except Exception as e:
        raise Exception(f"Failed to find comparables from database: {str(e)}")