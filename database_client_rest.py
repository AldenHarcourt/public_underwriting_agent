"""
REST API Database client for Supabase integration
Uses direct HTTP requests instead of the supabase Python library to avoid dependency issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import requests
import json
import os

# Handle Streamlit import for secrets
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class RestDatabaseClient:
    """Handles all database operations using direct REST API calls"""
    
    def __init__(self, url=None, key=None):
        """Initialize REST client using provided credentials or Streamlit secrets"""
        if url and key:
            self.url = url.rstrip('/')
            self.key = key
        elif STREAMLIT_AVAILABLE:
            self.url = st.secrets["supabase"]["url"].rstrip('/')
            self.key = st.secrets["supabase"]["anon_key"]
        else:
            self.url = os.getenv("SUPABASE_URL", "").rstrip('/')
            self.key = os.getenv("SUPABASE_ANON_KEY", "")
            if not self.url or not self.key:
                raise ValueError("No Supabase credentials found")
        
        self.headers = {
            'apikey': self.key,
            'Authorization': f'Bearer {self.key}',
            'Content-Type': 'application/json'
        }
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test the connection to Supabase"""
        try:
            response = requests.get(f"{self.url}/rest/v1/", headers=self.headers)
            if response.status_code != 200:
                raise Exception(f"Connection test failed with status {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to connect to Supabase: {str(e)}")
    
    def get_all_properties(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load all properties from the database with image URLs"""
        try:
            url = f"{self.url}/rest/v1/properties?select=*"
            if limit:
                url += f"&limit={limit}"
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"Query failed with status {response.status_code}: {response.text}")
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df = self._standardize_columns(df)
            df = self._convert_data_types(df)
            
            # Add image URLs to properties
            df = self._add_image_urls(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Database query failed: {str(e)}")
    
    def _query_bounding_box(
        self,
        lat_min: float,
        lat_max: float, 
        lon_min: float,
        lon_max: float,
        cutoff_date_str: str,
        property_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Query a specific bounding box for properties"""
        import time
        
        url = f"{self.url}/rest/v1/properties?select=*"
        url += f"&last_sold_date=gte.{cutoff_date_str}"
        url += f"&latitude=gte.{lat_min}&latitude=lte.{lat_max}"
        url += f"&longitude=gte.{lon_min}&longitude=lte.{lon_max}"
        url += f"&limit={limit}"
        
        if property_type:
            url += f"&property_type=eq.{property_type}"
        
        # Longer delay to prevent database overload
        time.sleep(0.5)
        
        try:
            response = requests.get(url, headers=self.headers, timeout=60)
            
            if response.status_code != 200:
                raise Exception(f"Query failed with status {response.status_code}: {response.text}")
            
            return response.json()
            
        except requests.exceptions.Timeout:
            raise Exception("Database query timed out - try reducing search area")
        except Exception as e:
            raise Exception(f"Database query failed: {str(e)}")

    def _recursive_spatial_search(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        cutoff_date_str: str,
        property_type: Optional[str] = None,
        max_depth: int = 3,
        current_depth: int = 0,
        query_count: int = 0
    ) -> List[Dict]:
        """Recursively search geographic area using quadtree approach"""
        
        # Circuit breaker: limit total number of queries
        if query_count >= 20:
            print(f"WARNING: Hit query limit ({query_count} queries) - stopping recursion")
            return []
        
        # Log the current search box
        box_size_lat = lat_max - lat_min
        box_size_lon = lon_max - lon_min
        print(f"DEBUG SPATIAL SEARCH: Depth {current_depth}, Query #{query_count+1}, Box size: {box_size_lat:.6f}° × {box_size_lon:.6f}°")
        
        # Query this bounding box
        try:
            results = self._query_bounding_box(
                lat_min, lat_max, lon_min, lon_max, 
                cutoff_date_str, property_type
            )
            query_count += 1
        except Exception as e:
            print(f"WARNING: Query failed at depth {current_depth}: {str(e)}")
            return []
        
        print(f"DEBUG SPATIAL SEARCH: Found {len(results)} properties in current box")
        
        # If we got less than 1000 results OR hit max depth OR box is too small, return what we have
        min_box_size = 0.01  # ~0.7 miles
        if (len(results) < 1000 or 
            current_depth >= max_depth or 
            box_size_lat < min_box_size or 
            box_size_lon < min_box_size):
            if len(results) == 1000 and current_depth >= max_depth:
                print(f"WARNING: Hit max depth {max_depth} but still getting 1000 results - some properties may be missed")
            return results
        
        # We hit the 1000 limit, so split into 4 quadrants
        print(f"DEBUG SPATIAL SEARCH: Hit 1000 limit, splitting into quadrants...")
        
        lat_mid = (lat_min + lat_max) / 2
        lon_mid = (lon_min + lon_max) / 2
        
        # Define the 4 quadrants: NW, NE, SW, SE
        quadrants = [
            ("NW", lat_min, lat_mid, lon_min, lon_mid),
            ("NE", lat_min, lat_mid, lon_mid, lon_max),
            ("SW", lat_mid, lat_max, lon_min, lon_mid),
            ("SE", lat_mid, lat_max, lon_mid, lon_max)
        ]
        
        all_results = []
        for quad_name, q_lat_min, q_lat_max, q_lon_min, q_lon_max in quadrants:
            print(f"DEBUG SPATIAL SEARCH: Searching {quad_name} quadrant...")
            quad_results = self._recursive_spatial_search(
                q_lat_min, q_lat_max, q_lon_min, q_lon_max,
                cutoff_date_str, property_type, max_depth, current_depth + 1, query_count
            )
            all_results.extend(quad_results)
            print(f"DEBUG SPATIAL SEARCH: {quad_name} quadrant returned {len(quad_results)} properties")
            
            # Update query count based on recursive calls
            query_count += 1  # Rough estimate - each quadrant uses at least 1 query
        
        return all_results

    def get_properties_by_location(
        self, 
        latitude: float, 
        longitude: float, 
        radius_miles: float = 10.0,
        max_age_months: int = 36,
        property_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Find properties within a geographic radius using smart recursive spatial search"""
        try:
            print(f"DEBUG SMART SEARCH: lat={latitude}, lon={longitude}, radius={radius_miles}, type={property_type}")
            
            # Calculate date cutoff
            cutoff_date = datetime.now() - timedelta(days=max_age_months * 30)
            cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
            print(f"DEBUG SMART SEARCH: Date cutoff={cutoff_date_str}")
            
            # Start with a much smaller geographic area to minimize recursion
            # Use radius_miles to determine initial search box size
            # 1 degree ≈ 69 miles, so convert radius to degrees
            degree_buffer = (radius_miles * 1.5) / 69.0  # 1.5x radius for small buffer
            degree_buffer = max(degree_buffer, 0.05)  # Minimum 0.05 degree buffer (~3.5 miles)
            degree_buffer = min(degree_buffer, 0.15)  # Maximum 0.15 degree buffer (~10 miles)
            
            lat_min = latitude - degree_buffer
            lat_max = latitude + degree_buffer
            lon_min = longitude - degree_buffer
            lon_max = longitude + degree_buffer
            
            print(f"DEBUG SMART SEARCH: Initial box: lat {lat_min:.6f} to {lat_max:.6f}, lon {lon_min:.6f} to {lon_max:.6f}")
            print(f"DEBUG SMART SEARCH: Box size: {degree_buffer*2:.6f}° ({degree_buffer*2*69:.1f} miles)")
            
            # Use recursive spatial search to get ALL properties in the area
            all_properties = self._recursive_spatial_search(
                lat_min, lat_max, lon_min, lon_max,
                cutoff_date_str, property_type
            )
            
            print(f"DEBUG SMART SEARCH: Total properties found: {len(all_properties)}")
            
            # Deduplicate by zpid (or id if zpid not available)
            seen_ids = set()
            unique_properties = []
            duplicate_count = 0
            
            for prop in all_properties:
                prop_id = prop.get('zpid') or prop.get('id')
                if prop_id and prop_id not in seen_ids:
                    seen_ids.add(prop_id)
                    unique_properties.append(prop)
                else:
                    duplicate_count += 1
            
            print(f"DEBUG SMART SEARCH: After deduplication: {len(unique_properties)} unique properties ({duplicate_count} duplicates removed)")
            
            # Check for our target property in raw results
            target_found = False
            for prop in unique_properties:
                if '4870' in str(prop.get('street_address', '')) and '39th' in str(prop.get('street_address', '')):
                    print(f"DEBUG SMART SEARCH: ✓ Found target property: {prop['street_address']}")
                    target_found = True
                    break
            if not target_found:
                print(f"DEBUG SMART SEARCH: ❌ Target property '4870 NE 39th Street' still not found")
            
            if not unique_properties:
                return pd.DataFrame()
            
            # Convert to DataFrame and standardize
            df = pd.DataFrame(unique_properties)
            df = self._standardize_columns(df)
            df = self._convert_data_types(df)
            
            # Add image URLs to properties
            df = self._add_image_urls(df)
            
            # Calculate exact distances and filter by radius
            if not df.empty:
                from geopy.distance import geodesic
                subject_coords = (latitude, longitude)
                
                df['distance_miles'] = df.apply(
                    lambda row: geodesic(
                        (row['latitude'], row['longitude']), 
                        subject_coords
                    ).miles if pd.notnull(row['latitude']) and pd.notnull(row['longitude']) else float('inf'), 
                    axis=1
                )
                
                # Filter by exact radius
                print(f"DEBUG SMART SEARCH: Before radius filter: {len(df)} properties")
                df = df[df['distance_miles'] <= radius_miles]
                print(f"DEBUG SMART SEARCH: After radius filter: {len(df)} properties within {radius_miles} miles")
                
                # Check if target property survived filtering
                target_in_final = False
                for _, row in df.iterrows():
                    if '4870' in str(row.get('streetAddress', '')) and '39th' in str(row.get('streetAddress', '')):
                        print(f"DEBUG SMART SEARCH: ✓ Target property in final results: {row['streetAddress']} at {row['distance_miles']:.3f} miles")
                        target_in_final = True
                        break
                if not target_in_final:
                    print(f"DEBUG SMART SEARCH: ❌ Target property not in final results after radius filter")
            
            return df
            
        except Exception as e:
            raise Exception(f"Smart spatial search failed: {str(e)}")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize database column names to match expected format"""
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
            'living_area': 'sqft',
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
    
    def _add_image_urls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add image URLs from property_images table to properties"""
        if df.empty:
            df['image_urls'] = pd.Series([], dtype=object)
            return df
        
        try:
            # Get all property IDs
            property_ids = df['id'].tolist() if 'id' in df.columns else []
            
            if not property_ids:
                df['image_urls'] = pd.Series([[] for _ in range(len(df))], dtype=object)
                return df
            
            # Query images for all properties
            # Use comma-separated list for the IN query
            ids_str = ','.join(map(str, property_ids))
            url = f"{self.url}/rest/v1/property_images?select=property_id,image_url,image_type&property_id=in.({ids_str})"
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                images_data = response.json()
                
                # Group images by property_id
                property_images = {}
                for img in images_data:
                    prop_id = img['property_id']
                    if prop_id not in property_images:
                        property_images[prop_id] = []
                    property_images[prop_id].append(img['image_url'])
                
                # Add image_urls column to dataframe
                df['image_urls'] = df['id'].apply(lambda pid: property_images.get(pid, []))
            else:
                # If image query fails, add empty lists
                df['image_urls'] = pd.Series([[] for _ in range(len(df))], dtype=object)
                
        except Exception as e:
            # If anything fails, add empty image lists
            df['image_urls'] = pd.Series([[] for _ in range(len(df))], dtype=object)
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types for proper analysis"""
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
_rest_db_client = None

def get_rest_database_client() -> RestDatabaseClient:
    """Get or create REST database client singleton"""
    global _rest_db_client
    if _rest_db_client is None:
        _rest_db_client = RestDatabaseClient()
    return _rest_db_client


def load_data_from_database(limit: Optional[int] = None) -> pd.DataFrame:
    """Load property data from database using REST API"""
    try:
        client = get_rest_database_client()
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
    """Find comparable properties using REST API"""
    try:
        lat = subject_property.get('latitude')
        lon = subject_property.get('longitude')
        
        if lat is None or lon is None:
            raise ValueError("Subject property missing latitude/longitude")
        
        property_type = subject_property.get('propertyType', 'singleFamily')
        
        client = get_rest_database_client()
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