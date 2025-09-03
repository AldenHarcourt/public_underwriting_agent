# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based AI Underwriting Agent for real estate valuation that uses comparative market analysis (CMA) to determine After Repair Value (ARV). **Database Version**: This version uses Supabase database instead of CSV files for improved performance and scalability.

## Key Commands

### Running the Application
```bash
# Development (local)
streamlit run app.py

# Production (Streamlit Cloud deployment)
# Automatic deployment from GitHub repository
# Uses .streamlit/secrets.toml for configuration
```

### Data Migration
```bash
# Migrate CSV data to Supabase database (one-time setup)
python migrate_csv_to_supabase.py

# Test migration (dry run)
python migrate_csv_to_supabase.py --dry-run

# Debug migration
python migrate_csv_to_supabase.py --debug
```

### Database Operations
```bash
# Test database connectivity
python -c "from database_client import get_database_client; client = get_database_client(); print('Database connected!')"
```

## Architecture Overview

### Core Components

**app.py** - Main Streamlit application
- Provides UI for property details input (manual or API-based)
- Handles user sessions and analysis state management
- Orchestrates the full underwriting workflow using database queries
- Displays results with interactive maps and comparable property selection

**agent_logic.py** - Core underwriting engine (Database Version)
- Property geocoding and address normalization
- Comparable property finding with optimized database queries
- Simplified AI-powered property adjustments (full AI features coming soon)
- Tiered filtering system for finding best comparables
- Geographic search optimization using PostGIS functions

**database_client.py** - Database integration layer
- Supabase client management and connection handling
- Optimized property queries with geographic filtering
- Data type conversion and column mapping
- Standardized interface for database operations

### Key Algorithms

**Comparable Selection Strategy**:
- Database-optimized geographic search within radius
- Tiered filtering: 12mo/0.5mi (Strict) → 18mo/1.0mi (Medium) → 24mo/2.0mi (Loose) → 36mo/3.0mi (Broad)
- Property type matching (singleFamily, multiFamily, land)
- Bed/bath similarity with rural property accommodation
- Square footage tolerance with diminishing returns model
- Lot size compatibility scoring for rural vs urban properties

**Valuation Methods**:
- Primary: Simplified price per square foot calculation
- Database-driven comp selection for accurate market analysis
- Lot size adjustments using context-aware land value analysis (urban/suburban/rural)
- Framework for AI adjustments (full implementation coming soon)

### Data Flow

1. **Input**: Property address + details (manual entry or RapidAPI fetch)
2. **Geocoding**: Convert address to lat/lon coordinates
3. **Database Query**: Find nearby properties using PostGIS geographic functions
4. **Comp Selection**: Apply tiered filtering to find 3-5 best comparable properties
5. **Valuation**: Calculate ARV using weighted comparable analysis
6. **Output**: ARV with confidence scoring and detailed comparable analysis

### Database Schema

**Properties Table** (`properties`):
- Core property data: location, address, price, bedrooms, bathrooms, living area
- Lot size with units, property type, sale dates
- Geographic indexing for fast proximity searches
- Tax assessment data and listing information

**Property Images Table** (`property_images`):
- Links to property images by type (main, streetview, satellite, all_photos)
- Foreign key relationship to properties table
- Support for multiple images per property

**Column Mapping**: Database columns are automatically mapped to expected format:
- `street_address` → `streetAddress`
- `living_area` → `sqft`
- `last_sold_date` → `dateSold`
- Full mapping handled in `database_client._standardize_columns()`

## Important Implementation Details

### Database Integration
- Uses Supabase PostgreSQL with PostGIS for geographic queries
- Row Level Security (RLS) enabled for data protection
- Optimized indexes for location, date, and property type queries
- Connection pooling and error handling for reliability

### OpenAI Integration
- API key stored in Streamlit secrets (`st.secrets["openai"]["api_key"]`)
- Simplified AI integration (full features coming in future updates)
- Graceful fallback when AI services unavailable

### Streamlit Cloud Deployment
- Configuration in `.streamlit/secrets.toml`
- Environment variables for database and API credentials
- Auto-deployment from GitHub repository
- Health monitoring and error logging

### Property Type Handling
- Rural properties (>1 acre): More lenient bed/bath matching, lot size critical
- Suburban (0.25-1 acre): Balanced criteria
- Urban (<0.25 acre): Square footage and location prioritized

### Performance Optimizations
- Database queries instead of loading all data into memory
- Geographic indexing for fast proximity searches
- Streamlit caching for database connections
- Batch processing for data migration

## Database Migration Notes

- **One-time setup**: Run `migrate_csv_to_supabase.py` to import CSV data
- **Data validation**: Automatic cleaning and type conversion during migration
- **Error handling**: Comprehensive logging and recovery for failed imports
- **Batch processing**: Configurable batch sizes for large datasets

## Development Notes

- Pure Python project with requirements.txt
- Supabase client for database operations with geographic search
- Streamlit session state for UI persistence across analysis runs
- Error handling for database connectivity and missing data
- Debug modes available in database client and migration scripts
- Simplified initial version - full AI features planned for future releases

## Migration from CSV Version

This database version maintains API compatibility with the original CSV-based version:
- Same function names and return formats
- Same UI and user experience  
- Same analysis algorithms with database-optimized queries
- Improved performance and scalability