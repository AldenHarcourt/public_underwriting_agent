# AI Underwriting Agent Setup Instructions

## Overview
This is a complete Streamlit-based AI underwriting agent with Supabase database integration and property image analysis capabilities.

## Step 1: Database Setup

1. **Execute Database Schema**
   - Open your Supabase dashboard at https://supabase.com/dashboard
   - Go to your project and navigate to "SQL Editor"
   - Copy and paste the contents of `database_schema.sql`
   - Click "Run" to execute the schema

2. **Verify Tables Created**
   - Go to "Table Editor" in Supabase dashboard
   - You should see `properties` and `property_images` tables
   - Check that indexes are created (Database > Indexes)

## Step 2: Data Migration (COMPLETED)

✅ **Property Data**: Already migrated to Supabase database
✅ **Property Images**: 762k+ images migrated successfully  
✅ **Database Integration**: Fully functional

*Note: If you need to re-migrate images, use `migrate_images_ultra_fast.py --live`*

## Step 3: Application Configuration

1. **Configure Secrets**
   Create `.streamlit/secrets.toml` with:
   ```toml
   [supabase]
   url = "your_supabase_url"
   anon_key = "your_supabase_anon_key"

   [openai]
   api_key = "your_openai_api_key"

   [rapidapi]
   key = "your_rapidapi_key_optional"
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test Application Locally**
   ```bash
   streamlit run app.py
   ```

## Step 4: Streamlit Cloud Deployment

1. **Prepare Repository**
   - Push this directory to GitHub
   - Ensure `.gitignore` excludes sensitive files

2. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Connect your GitHub repository
   - Select this directory as the app folder
   - Configure secrets in the Streamlit Cloud dashboard

3. **Verify Deployment**
   - Test property search and analysis
   - Confirm image analysis is working
   - Check comparable property functionality

## Features Included

✅ **Property Data Input**: Manual entry and API integration  
✅ **Database Integration**: Supabase with 350k+ properties  
✅ **Image Analysis**: AI-powered property image analysis  
✅ **Comparable Analysis**: Geographic search with image integration  
✅ **Interactive UI**: Maps, property selection, ARV calculation  
✅ **Production Ready**: Optimized for Streamlit Cloud deployment  

## Database Status

- **Properties**: 346k+ properties in database
- **Images**: 762k+ property images available  
- **Geographic Search**: Optimized PostGIS queries
- **Image Integration**: Automatic image loading for AI analysis

## Troubleshooting

- **No images showing**: Verify property_images table has data
- **Slow performance**: Check database query optimization
- **AI analysis fails**: Verify OpenAI API key in secrets
- **Database connection**: Check Supabase credentials

## Architecture

- **Frontend**: Streamlit application (`app.py`)
- **Backend Logic**: Core algorithms (`agent_logic.py`) 
- **Database**: Supabase client (`database_client_rest.py`)
- **Schema**: PostgreSQL with PostGIS (`database_schema.sql`)