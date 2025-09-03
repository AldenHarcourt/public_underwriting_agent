# Database Version Setup Instructions

## Step 1: Database Setup (YOU MUST DO)

1. **Execute Database Schema**
   - Open your Supabase dashboard at https://supabase.com/dashboard
   - Go to your project: yqqsdphruymqdckuzjzh
   - Navigate to "SQL Editor"
   - Copy and paste the contents of `database_schema.sql`
   - Click "Run" to execute the schema

2. **Verify Tables Created**
   - Go to "Table Editor" in Supabase dashboard
   - You should see `properties` and `property_images` tables
   - Check that indexes are created (visible in Database > Indexes)

## Step 2: Data Migration (AFTER I CREATE THE SCRIPT)

1. **Install Python Dependencies**
   ```bash
   pip install supabase pandas python-dotenv
   ```

2. **Run Migration Script**
   ```bash
   python migrate_csv_to_supabase.py
   ```
   - This will upload your CSV data to Supabase
   - Monitor progress in the terminal

## Step 3: Application Setup

1. **Update Secrets**
   - Edit `.streamlit/secrets.toml`
   - Add your OpenAI API key
   - Add your RapidAPI key (if using)

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test Locally**
   ```bash
   streamlit run app.py
   ```

## Step 4: Streamlit Cloud Deployment (YOU MUST DO)

1. **Push to GitHub**
   - Create GitHub repository
   - Push this `database_version` folder as root

2. **Connect Streamlit Cloud**
   - Go to share.streamlit.io
   - Connect your GitHub repo
   - Deploy from main branch

3. **Configure Secrets in Streamlit Cloud**
   - In Streamlit Cloud dashboard, go to app settings
   - Add secrets from `.streamlit/secrets.toml`
   - Deploy the app

## Current Status
- ✅ Directory structure created
- ✅ Database schema ready
- ✅ Secrets template configured
- ⏳ Next: Core application files migration
- ⏳ Next: Data migration script
- ⏳ Next: Database integration code