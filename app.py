import streamlit as st
import pandas as pd
import agent_logic  # Import our refactored logic

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Underwriting Agent",
    page_icon="🤖",
    layout="wide"
)

# --- App State Management ---
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'final_arv' not in st.session_state:
    st.session_state.final_arv = 0
if 'report_str' not in st.session_state:
    # Initialize empty DataFrames for comps if they don't exist
    if 'selected_comps' not in st.session_state:
        st.session_state.selected_comps = pd.DataFrame()
    if 'backup_comps' not in st.session_state:
        st.session_state.backup_comps = pd.DataFrame()
    
    # Ensure selected_comps is always a DataFrame, not None
    selected_comps = st.session_state.selected_comps if 'selected_comps' in st.session_state and st.session_state.selected_comps is not None else pd.DataFrame()
    backup_comps = st.session_state.backup_comps if 'backup_comps' in st.session_state and st.session_state.backup_comps is not None else pd.DataFrame()
if 'adjusted_comps' not in st.session_state:
    st.session_state.adjusted_comps = pd.DataFrame()
if 'subject_images' not in st.session_state:
    st.session_state.subject_images = []
if 'error_message' not in st.session_state:
    st.session_state.error_message = ""

# --- Data Loading ---
@st.cache_data
def load_data():
    """Load data from Supabase database"""
    try:
        if not hasattr(agent_logic, 'load_data_from_database'):
            raise ImportError('agent_logic.load_data_from_database is not defined')
        df = agent_logic.load_data_from_database()
        return df
    except Exception as e:
        st.error(f"Database Error: {e}. Please check your database connection.")
        return None

all_comps_df = load_data()

# --- UI Sidebar for Inputs ---
with st.sidebar:
    st.title("🤖 AI Underwriting Agent")
    st.header("Subject Property Details")

    # Input mode selection
    input_mode = st.radio(
        "Input Mode",
        ["API-First (Recommended)", "Manual Entry Only"],
        help="API-First: Enter address and let the system fetch property details. Manual: Enter all details yourself."
    )

    # API Key input (always shown but only required for API mode)
    api_key = st.text_input(
        "RapidAPI Key",
        value="",
        type="password",
        help="Required for API-First mode. Optional for Manual mode."
    )

    if input_mode == "API-First (Recommended)":
        st.info("Enter the property address and the system will automatically fetch details from the database.")
        
        # Address inputs for API mode
        subject_address = st.text_input("Street Address", "4701 NE 39th St", help="Include the full street number and name, e.g. '4701 NE 39th St'")
        subject_city = st.text_input("City", "Seattle")
        subject_state = st.text_input("State", "WA")
        subject_zipcode = st.text_input("Zip Code", "98105")
        
        # Manual override fields (hidden by default, can be expanded if needed)
        with st.expander("Manual Override (if API data is incorrect)"):
            st.warning("Only use if the API data is incorrect. Leave empty to use API data.")
            beds = st.number_input("Bedrooms (override)", min_value=0, max_value=10, value=None, step=1, help="Leave empty to use API data")
            baths = st.number_input("Bathrooms (override)", min_value=0.0, max_value=10.0, value=None, step=0.5, help="Leave empty to use API data")
            sqft = st.number_input("Square Footage (override)", min_value=100, max_value=10000, value=None, step=50, help="Leave empty to use API data")
            property_type = st.selectbox(
                "Property Type (override)",
                ["", "singleFamily", "multiFamily", "land"],
                help="Leave empty to use API data"
            )
    
    else:  # Manual Entry Only
        st.info("Manual entry mode. You must provide all property details. API calls will not be made.")
        
        # Address inputs for manual mode
        subject_address = st.text_input("Street Address", "4701 NE 39th St", help="Include the full street number and name, e.g. '4701 NE 39th St'")
        subject_city = st.text_input("City", "Seattle")
        subject_state = st.text_input("State", "WA")
        subject_zipcode = st.text_input("Zip Code", "98105")
        
        # Required manual inputs
        beds = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
        baths = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        sqft = st.number_input("Square Footage", min_value=100, max_value=10000, value=860, step=50)
        property_type = st.selectbox(
            "Property Type",
            ["singleFamily", "multiFamily", "land"],
            help="Select the type of property for accurate comparable selection"
        )

    st.header("Analysis Configuration")
    num_runs = st.slider(
        "Number of Analysis Runs", 
        min_value=1, 
        max_value=5, 
        value=1,
        step=1, 
        help="Select the number of ARV calculations to perform and average."
    )

    run_button = st.button(
        "Run Underwriting Analysis",
        type="primary",
        use_container_width=True,
        disabled=(all_comps_df is None)
    )

# --- Main Page for Results ---
st.title("Underwriting Analysis Results")

if run_button:
    # Reset all relevant session state variables when starting a new analysis
    st.session_state.analysis_complete = False
    st.session_state.error_message = ""
    # Combine selected and backup comps for display
    selected_comps = st.session_state.selected_comps if 'selected_comps' in st.session_state and st.session_state.selected_comps is not None else pd.DataFrame()
    backup_comps = st.session_state.backup_comps if 'backup_comps' in st.session_state and st.session_state.backup_comps is not None else pd.DataFrame()
    all_comps = pd.concat([selected_comps, backup_comps]) if not backup_comps.empty else selected_comps.copy()
    
    # Initialize subject property with address info
    subject_property = {}
    subject_property['streetAddress'] = str(subject_address)
    subject_property['city'] = str(subject_city)
    subject_property['state'] = str(subject_state)
    subject_property['zipcode'] = str(subject_zipcode)
    subject_property_source = "manual"

    # Handle API-First mode
    if input_mode == "API-First (Recommended)":
        if not api_key:
            st.session_state.error_message = "You must enter your RapidAPI key to use API-First mode."
        else:
            try:
                # Try to fetch property details from API
                api_property = agent_logic.fetch_subject_property_from_api(subject_property, api_key)
                subject_property = api_property
                subject_property_source = "api"
                
                # Apply manual overrides if provided
                if beds is not None:
                    subject_property['bedrooms'] = int(beds)
                if baths is not None:
                    subject_property['bathrooms'] = float(baths)
                if sqft is not None:
                    subject_property['sqft'] = int(sqft)
                if property_type:
                    subject_property['propertyType'] = str(property_type)
                    
            except Exception as e:
                st.session_state.error_message = f"API fetch failed: {e}. Please try Manual Entry Only mode or check your API key."
                # Fall back to manual entry with just address
                subject_property['bedrooms'] = int(beds) if beds is not None else 0
                subject_property['bathrooms'] = float(baths) if baths is not None else 1.0
                subject_property['sqft'] = int(sqft) if sqft is not None else 1000
                subject_property['propertyType'] = str(property_type) if property_type else 'singleFamily'
    
    else:  # Manual Entry Only mode
        # Use manually entered values
        subject_property['bedrooms'] = int(beds) if beds is not None else 0
        subject_property['bathrooms'] = float(baths) if baths is not None else 1.0
        subject_property['sqft'] = int(sqft) if sqft is not None else 1000
        subject_property['propertyType'] = str(property_type) if property_type else 'singleFamily'

    with st.spinner(f"Starting {num_runs}-run analysis... This may take a few minutes."):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        progress_tracker = [0.0]
        # More detailed progress calculation
        TOTAL_STEPS = 3 + (num_runs * (1 + 5 + 1 + 1)) + 1 # Geocode, Zillow Search, Subject Img Desc + (Find, Img, Adj, Calc) * runs + Report
        PROGRESS_INCREMENT = 1.0 / TOTAL_STEPS if TOTAL_STEPS > 0 else 1.0

        def update_progress(message):
            status_text.text(message)
            progress_tracker[0] += PROGRESS_INCREMENT
            progress_bar.progress(min(progress_tracker[0], 1.0))

        try:
            result = agent_logic.run_full_analysis(
                subject_property,
                all_comps_df,
                num_runs,
                progress_callback=update_progress
            )
            st.session_state.final_arv = result['best_arv']
            st.session_state.arv_range = result['arv_range']
            st.session_state.arv_iqr = result['arv_iqr']
            st.session_state.confidence = result['confidence']
            st.session_state.all_arvs = result['all_arvs']
            st.session_state.report_str = result['summary_report']
            # Store the final comps and backup comps in session state
            st.session_state.adjusted_comps = result['final_comps_df']
            st.session_state.selected_comps = result['final_comps_df'].copy()  # Initialize selected comps
            st.session_state.subject_images = result['subject_images']
            st.session_state.subject_property = subject_property  # Store subject_property in session state
            st.session_state.backup_comps = result.get('backup_comps', pd.DataFrame())
            st.session_state.removed_comps = []  # Track removed comps for re-adding
            st.session_state.analysis_complete = True
            st.session_state.subject_property_source = subject_property_source
            status_text.success("Analysis complete!")
            progress_bar.progress(1.0)

        except (ValueError, FileNotFoundError) as e:
            st.session_state.error_message = str(e)
            progress_bar.empty(); status_text.empty()
        except Exception as e:
            st.session_state.error_message = f"An unexpected error occurred: {e}"
            progress_bar.empty(); status_text.empty()

# --- Display Results or Welcome Message ---
if st.session_state.error_message:
    st.error(st.session_state.error_message)

if st.session_state.analysis_complete:
    # Retrieve subject_property from session state for use in this section
    subject_property = st.session_state.get('subject_property', {})
    if not isinstance(subject_property, dict):
        subject_property = {}
    sp = subject_property  # Always a dict now
    # ARV header and subject property image side by side
    arv_col, image_col = st.columns([2, 1])
    
    with arv_col:
        if st.session_state.get('all_arvs') and len(st.session_state.all_arvs) == 1:
            st.markdown(
                f"<h1 style='color:#2e7bcf; font-size: 3em;'>ARV: ${st.session_state.final_arv:,.0f}</h1>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h1 style='color:#2e7bcf; font-size: 3em;'>ARV: ${st.session_state.final_arv:,.0f}</h1>",
                unsafe_allow_html=True
            )
            min_arv, max_arv = st.session_state.arv_range
            iqr_low, iqr_high = st.session_state.arv_iqr
            st.markdown(f"<b>ARV Range:</b> ${min_arv:,.0f} - ${max_arv:,.0f}", unsafe_allow_html=True)
            st.markdown(f"<b>Interquartile Range (IQR):</b> ${iqr_low:,.0f} - ${iqr_high:,.0f}", unsafe_allow_html=True)
    
    with image_col:
        # Display subject property image if available
        subject_images = sp.get('image_urls', [])
        if subject_images and len(subject_images) > 0:
            st.markdown("**Subject Property**")
            try:
                st.image(subject_images[0], caption=f"{sp.get('streetAddress', 'Subject Property')}", width=300)
            except Exception as e:
                st.write("📷 *Image not available*")
        else:
            st.write("📷 *No subject images available*")

    # --- Show the actual selected comps table ---
    st.subheader("Top 5 Selected Comparables Used in Analysis")
    if not st.session_state.adjusted_comps.empty:
        df = st.session_state.adjusted_comps.copy()
        
        # Add formatted lot size column if the required columns exist
        if 'lot_size' in df.columns and 'lot_size_unit' in df.columns:
            df['lot_size_display'] = df.apply(
                lambda x: f"{x['lot_size']} {x['lot_size_unit']}" 
                if pd.notnull(x['lot_size']) and pd.notnull(x['lot_size_unit'])
                else 'N/A',
                axis=1
            )
        else:
            df['lot_size_display'] = 'N/A'
        
        # Define columns to display (updated for database schema)
        display_cols = [
            'streetAddress', 'city', 'state', 'zipcode',
            'bedrooms', 'bathrooms', 'sqft', 'lot_size_display',
            'price', 'distance_miles', 'dateSold', 'comp_tier'
        ]
        
        # Display the dataframe with formatted columns
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols].head(5))
    else:
        st.info("No comps available for display.")

    # --- Interactive Comp Review ---
    # Ensure we have the necessary session state variables
    if 'selected_comps' not in st.session_state:
        st.session_state.selected_comps = st.session_state.adjusted_comps.copy()
    if 'removed_comps' not in st.session_state:
        st.session_state.removed_comps = []

    selected_comps = st.session_state.selected_comps
    backup_comps = st.session_state.get('backup_comps', pd.DataFrame())
    removed_comps = st.session_state.get('removed_comps', [])

    # --- Review and Edit Comparables ---
    st.markdown("### Review and Edit Comparables")
    remove_indices = []
    # Mapping from comp_tier string to label
    tier_label_map = {
        "12mo_0.5mi": "Strict",
        "18mo_1.0mi": "Medium", 
        "24mo_2.0mi": "Loose",
        "36mo_3.0mi": "Broad",
        # New direct mapping (agent_logic now assigns these directly)
        "Strict": "Strict",
        "Medium": "Medium",
        "Loose": "Loose", 
        "Broad": "Broad"
    }
        
    for idx, row in selected_comps.iterrows():
        with st.container():
            tier_label = tier_label_map.get(str(row.get('comp_tier', '')), str(row.get('comp_tier', '')))
            dist = row.get('distance_miles', None)
            dist_str = f" | {dist:.2f} mi away" if dist is not None and pd.notnull(dist) else ""
            
            # Handle lot size display
            lot_size = row.get('lot_size', '')
            lot_size_unit = row.get('lot_size_unit', '')
            
            # Convert to string for safe comparison and handle different types
            lot_size_str = str(lot_size).strip() if lot_size is not None else ''
            lot_size_unit_str = str(lot_size_unit).strip() if lot_size_unit is not None else ''
            
            # Build lot size string if we have valid data
            if lot_size_str and lot_size_str.lower() != 'nan' and lot_size_str != 'None':
                if lot_size_unit_str and lot_size_unit_str.lower() not in ['', 'nan', 'none']:
                    lot_size_str = f" | Lot: {lot_size_str} {lot_size_unit_str}"
                else:
                    lot_size_str = f" | Lot: {lot_size_str}"
            else:
                lot_size_str = ""
            
            # Format sold date
            sold_date = row.get('dateSold', '')
            if pd.notna(sold_date) and sold_date != '':
                try:
                    if isinstance(sold_date, str):
                        sold_date = pd.to_datetime(sold_date)
                    sold_date_str = f" | Sold: {sold_date.strftime('%m/%d/%Y')}"
                except:
                    sold_date_str = f" | Sold: {sold_date}"
            else:
                sold_date_str = ""
            
            # Use database column names
            address = row.get('streetAddress', 'N/A')
            price = row.get('price', 0)
            bedrooms = row.get('bedrooms', 0)
            bathrooms = row.get('bathrooms', 0)
            living_area = row.get('sqft', 0)
            
            st.write(f"**{address}** | ${price:,.0f} | {bedrooms}bd/{bathrooms}ba | {living_area} sqft{lot_size_str} | {tier_label}{dist_str}{sold_date_str}")
            # Create unique key using row index and address to avoid duplicates
            unique_key = f"remove_{idx}_{hash(address) % 10000}"
            if st.button(f"Remove", key=unique_key):
                remove_indices.append(idx)
    
    # Remove selected comps
    if remove_indices:
        for idx in remove_indices:
            removed_row = selected_comps.loc[idx]
            st.session_state.removed_comps.append(removed_row)
        st.session_state.selected_comps = selected_comps.drop(remove_indices).reset_index(drop=True)
        # Coerce columns to numeric after removal
        for col in ['sqft', 'price', 'ai_adjustment', 'ai_weight']:
            if col in st.session_state.selected_comps.columns:
                st.session_state.selected_comps[col] = pd.to_numeric(st.session_state.selected_comps[col], errors='coerce')
        st.rerun()

    # --- Add Comparables Section ---
    st.subheader("Add Comparables")
    
    # Build addable comps pool (backup comps + removed comps, excluding already selected)
    selected_addresses = set(st.session_state.selected_comps['streetAddress'])
    addable_rows = []

    # Add backup comps
    if not backup_comps.empty:
        for _, row in backup_comps.iterrows():
            if row['streetAddress'] not in selected_addresses:
                addable_rows.append(row)

    # Add removed comps (if not already in backup or selected)
    backup_addresses = set(backup_comps['streetAddress']) if not backup_comps.empty else set()
    for row in removed_comps:
        if row['streetAddress'] not in selected_addresses and row['streetAddress'] not in backup_addresses:
            addable_rows.append(row)

    # Build addable comps with descriptive blurbs for the dropdown
    addable_comps_with_blurbs = []
    addable_comps_lookup = {}  # Map from blurb back to row data
    
    if addable_rows:
        for row in addable_rows:
            address = row['streetAddress']
            if address not in selected_addresses:
                # Create descriptive blurb similar to the review section
                price = f"${row.get('price', 0):,.0f}" if row.get('price', 0) > 0 else "N/A"
                beds = row.get('bedrooms', 'N/A')
                baths = row.get('bathrooms', 'N/A')
                sqft = row.get('sqft', 'N/A')
                
                # Handle lot size display (using same logic as review section)
                lot_size = row.get('lot_size', '')
                lot_size_unit = row.get('lot_size_unit', '')
                
                # Convert to string for safe comparison and handle different types
                lot_size_str = str(lot_size).strip() if lot_size is not None else ''
                lot_size_unit_str = str(lot_size_unit).strip() if lot_size_unit is not None else ''
                
                # Build lot size string if we have valid data
                if lot_size_str and lot_size_str.lower() != 'nan' and lot_size_str != 'None':
                    if lot_size_unit_str and lot_size_unit_str.lower() not in ['', 'nan', 'none']:
                        lot_size_display = f" | Lot: {lot_size_str} {lot_size_unit_str}"
                    else:
                        lot_size_display = f" | Lot: {lot_size_str}"
                else:
                    lot_size_display = ""
                
                # Get distance if available
                dist = row.get('distance_miles', None)
                dist_str = f" | {dist:.2f} mi" if dist is not None and pd.notnull(dist) else ""
                
                # Get comp tier if available
                tier = row.get('comp_tier', '')
                tier_label = tier_label_map.get(str(tier), str(tier))
                tier_str = f" | {tier_label}" if tier_label else ""
                
                # Format sold date for addable comp blurb
                sold_date = row.get('dateSold', '')
                if pd.notna(sold_date) and sold_date != '':
                    try:
                        if isinstance(sold_date, str):
                            sold_date = pd.to_datetime(sold_date)
                        sold_date_str = f" | Sold: {sold_date.strftime('%m/%d/%Y')}"
                    except:
                        sold_date_str = f" | Sold: {sold_date}"
                else:
                    sold_date_str = ""
                
                # Create the descriptive blurb
                blurb = f"{address} | {price} | {beds}bd/{baths}ba | {sqft} sqft{lot_size_display}{dist_str}{tier_str}{sold_date_str}"
                
                addable_comps_with_blurbs.append(blurb)
                addable_comps_lookup[blurb] = row

    # Remove any comps already selected (additional safety check)
    selected_addresses = set(st.session_state.selected_comps['streetAddress'])
    final_addable_comps = []
    final_lookup = {}
    for blurb, row_data in addable_comps_lookup.items():
        if row_data['streetAddress'] not in selected_addresses:
            final_addable_comps.append(blurb)
            final_lookup[blurb] = row_data

    # Display the dropdown for adding comparables
    if final_addable_comps:
        selected_comp_blurb = st.selectbox(
            "Choose a comparable to add:", 
            [""] + final_addable_comps, 
            key="add_comp_selectbox"
        )
        
        if selected_comp_blurb and selected_comp_blurb in final_lookup:
            if st.button("Add Selected Comparable", key="add_comp_button"):
                new_comp = final_lookup[selected_comp_blurb]
                # Convert to DataFrame row and add to selected comps
                new_comp_df = pd.DataFrame([new_comp])
                st.session_state.selected_comps = pd.concat([st.session_state.selected_comps, new_comp_df], ignore_index=True)
                # Remove from removed_comps if it was there
                st.session_state.removed_comps = [comp for comp in st.session_state.removed_comps 
                                                if comp['streetAddress'] != new_comp['streetAddress']]
                st.rerun()
    else:
        st.info("No additional comparables available to add.")

    # --- Recompute ARV Button ---
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Recompute ARV with Current Comps", key="recompute_arv", type="primary"):
        # Recalculate ARV with current comp selection
        if not st.session_state.selected_comps.empty:
            try:
                # Simple recalculation using current comp selection
                valid_comps = st.session_state.selected_comps[
                    (st.session_state.selected_comps['price'] > 0) & 
                    (st.session_state.selected_comps['sqft'] > 0) & 
                    pd.notnull(st.session_state.selected_comps['price']) & 
                    pd.notnull(st.session_state.selected_comps['sqft'])
                ]
                
                if not valid_comps.empty:
                    # Apply OpenAI analysis for recalculation
                    from agent_logic import get_structured_adjustments
                    
                    adjustment_result = get_structured_adjustments(sp, valid_comps)
                    adjustment_map = adjustment_result['adjustments']
                    arv_summary = adjustment_result.get('arv_summary', 'ARV recalculated using AI-enhanced weighted analysis.')
                    
                    # Apply AI adjustments
                    def safe_get(addr, field, default):
                        return adjustment_map.get(addr, {}).get(field, default)
                    
                    valid_comps['ai_adjustment'] = valid_comps['streetAddress'].apply(lambda addr: safe_get(addr, 'adjustment', 0))
                    valid_comps['ai_weight'] = valid_comps['streetAddress'].apply(lambda addr: safe_get(addr, 'weight', 1.0 / len(valid_comps)))
                    valid_comps['ai_explanation'] = valid_comps['streetAddress'].apply(lambda addr: safe_get(addr, 'explanation', 'No explanation available'))
                    
                    # Apply adjustments and calculate weighted average
                    valid_comps['adjusted_price'] = valid_comps['price'] + valid_comps['ai_adjustment']
                    valid_comps['price_per_sqft'] = valid_comps['adjusted_price'] / valid_comps['sqft']
                    
                    if valid_comps['ai_weight'].sum() > 0:
                        avg_price_per_sqft = (
                            (valid_comps['price_per_sqft'] * valid_comps['ai_weight']).sum() / 
                            valid_comps['ai_weight'].sum()
                        )
                    else:
                        avg_price_per_sqft = valid_comps['price_per_sqft'].mean()
                    
                    subject_sqft = sp.get('sqft', 1000)
                    if subject_sqft <= 0:
                        subject_sqft = 1000
                        
                    new_arv = avg_price_per_sqft * subject_sqft
                    
                    # Update session state
                    st.session_state.final_arv = int(new_arv)
                    st.session_state.all_arvs = [int(new_arv)]
                    st.session_state.adjusted_comps = st.session_state.selected_comps.copy()
                    
                    # Format subject property details
                    full_address = f"{sp.get('streetAddress', 'Unknown')}, {sp.get('city', 'Unknown')}, {sp.get('state', 'Unknown')}, {sp.get('zipcode', 'Unknown')}"
                    subject_sqft = sp.get('sqft', 0)
                    lot_size_display = sp.get('lotSize', 'N/A')
                    if sp.get('lotSizeUnit'):
                        lot_size_display += f" {sp.get('lotSizeUnit')}"
                    
                    # Generate new summary report using consistent formatting
                    new_report = f"""--- Underwriting Summary Report (1-Run Average) ---
Subject Property: {full_address}
  - {sp.get('bedrooms', 'N/A')} bed, {sp.get('bathrooms', 'N/A')} bath, {subject_sqft} sqft

Run 1 ARV: ${int(new_arv):,.2f}

--- FINAL AVERAGED AFTER-REPAIR VALUE (ARV): ${int(new_arv):,.2f} ---


--- Detailed Run-by-Run Analysis ---

--- Individual Run ARV: ${int(new_arv):,.2f} ---
Subject Property: {full_address}
  Beds: {sp.get('bedrooms', 'N/A')}, Baths: {sp.get('bathrooms', 'N/A')}, Sqft: {subject_sqft:,}, Lot Size: {lot_size_display} 
  Calculated ARV: ${int(new_arv):,.2f}, Price/Sqft: ${avg_price_per_sqft:.2f}
  Image Analysis: {sp.get('image_description', 'No image analysis available for subject property.')}

ARV Calculation Summary: {arv_summary}

Valuation Method: AI-Enhanced Weighted Analysis (Recalculated)
AI-Weighted Average Price/Sqft: ${avg_price_per_sqft:.2f}

{agent_logic.generate_comparable_summary_table(valid_comps)}

Note: This analysis uses selected comparables for manual ARV recalculation.
"""
                    st.session_state.report_str = new_report
                    
                    st.success(f"ARV recalculated! New estimate: ${int(new_arv):,}")
                    st.rerun()
                else:
                    st.error("No valid comparables with price and sqft data for recalculation.")
            except Exception as e:
                st.error(f"Error recalculating ARV: {str(e)}")
        else:
            st.error("No comparables selected for recalculation.")

    # --- Map Visualization ---
    # Build map data with subject property and all available comps
    map_rows = []
    
    # Add subject property
    if 'latitude' in sp and 'longitude' in sp and not (pd.isna(sp['latitude']) or pd.isna(sp['longitude'])):
        map_rows.append({
            'latitude': sp['latitude'],
            'longitude': sp['longitude'],
            'type': 'subject',
            'streetAddress': sp.get('streetAddress', 'Subject Property'),
            'price': sp.get('price', 0),
            'bedrooms': sp.get('bedrooms', 0),
            'bathrooms': sp.get('bathrooms', 0),
            'sqft': sp.get('sqft', 0),
            'lot_size': sp.get('lot_size', ''),
            'lot_size_unit': sp.get('lot_size_unit', '')
        })
    
    # Add selected comps (blue dots)
    if not st.session_state.selected_comps.empty:
        selected_comps_copy = st.session_state.selected_comps.copy()
        selected_comps_copy['type'] = 'selected'
        map_rows.extend(selected_comps_copy.to_dict('records'))
    
    # Add addable comps (yellow dots) - backup comps + removed comps not currently selected
    if addable_rows:
        yellow_df = pd.DataFrame(addable_rows)
        yellow_df['type'] = 'addable'
        map_rows.extend(yellow_df.to_dict('records'))

    # Create map_df from collected rows
    if map_rows:
        map_df = pd.DataFrame(map_rows)
        
        # Add tooltip info for each point
        def tooltip_blurb(row):
            addr = row.get('streetAddress', 'N/A')
            if row['type'] == 'subject':
                beds = sp.get('bedrooms', 'N/A')
                baths = sp.get('bathrooms', 'N/A')
                sqft = sp.get('sqft', 'N/A')
                return f"🏠 SUBJECT: {addr} | {beds}bd/{baths}ba | {sqft} sqft"
            else:
                price = f"${row.get('price', 0):,.0f}" if row.get('price', 0) > 0 else "N/A"
                beds = row.get('bedrooms', 'N/A')
                baths = row.get('bathrooms', 'N/A')
                sqft = row.get('sqft', 'N/A')
                dist = row.get('distance_miles', None)
                dist_str = f" | {dist:.2f} mi" if dist is not None and pd.notnull(dist) else ""
                comp_type = "SELECTED" if row['type'] == 'selected' else "AVAILABLE"
                return f"🏘️ {comp_type}: {addr} | {price} | {beds}bd/{baths}ba | {sqft} sqft{dist_str}"
        
        map_df['tooltip'] = map_df.apply(tooltip_blurb, axis=1)
        
        # Create the map using pydeck
        if isinstance(map_df, pd.DataFrame) and map_df[['latitude', 'longitude']].notnull().to_numpy().all():
            import pydeck as pdk
            
            def color_picker(t):
                if t == 'subject':
                    return [200, 30, 0, 160]  # Red
                elif t == 'selected':
                    return [0, 0, 200, 160]   # Blue
                elif t == 'addable':
                    return [230, 200, 0, 180] # Yellow
                else:
                    return [100, 100, 100, 100]
            
            map_df['color'] = map_df['type'].apply(color_picker)
            
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[longitude, latitude]',
                get_color='color',
                get_radius=40,  # Smaller dots
                pickable=False
            )
            
            text_layer = pdk.Layer(
                "TextLayer",
                data=map_df,
                get_position='[longitude, latitude]',
                get_text='streetAddress',
                get_color=[255, 255, 255, 220],  # White text, mostly opaque
                get_size=10,
                get_alignment_baseline="'bottom'",
                get_pixel_offset="[0, -20]"
            )
            
            view_state = pdk.ViewState(
                latitude=map_df['latitude'].mean(),
                longitude=map_df['longitude'].mean(),
                zoom=14,  # More zoomed in
                pitch=0
            )
            
            r = pdk.Deck(
                layers=[scatter_layer, text_layer],
                initial_view_state=view_state
            )
            
            st.markdown("### Map of Subject and Comparables")
            st.pydeck_chart(r)
            
            # Add a legend
            st.markdown(
                '<div style="display: flex; gap: 2em; align-items: center; margin-bottom: 1em;"><span style="display: flex; align-items: center;"><span style="background: #C81E00; width: 18px; height: 18px; border-radius: 50%; display: inline-block; margin-right: 6px; border: 1px solid #888;"></span> Subject Property</span><span style="display: flex; align-items: center;"><span style="background: #0032C8; width: 18px; height: 18px; border-radius: 50%; display: inline-block; margin-right: 6px; border: 1px solid #888;"></span> Selected Comp</span><span style="display: flex; align-items: center;"><span style="background: #E6C800; width: 18px; height: 18px; border-radius: 50%; display: inline-block; margin-right: 6px; border: 1px solid #888;"></span> Addable Comp</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.info("No valid latitude/longitude for subject or comps. Map not shown.")
    else:
        st.info("No data available for map display.")

    # --- Confidence Score ---
    if not st.session_state.adjusted_comps.empty:
        df = st.session_state.adjusted_comps
        tier_numeric = {'Strict': 4, 'Medium': 3, 'Loose': 2, 'Broad': 1}
        if 'comp_tier' in df.columns:
            comp_tier_labels = df['comp_tier'].map(lambda t: tier_label_map.get(str(t), str(t)))
            comp_tier_scores = comp_tier_labels.map(lambda t: tier_numeric.get(t, 1))
            avg_score = comp_tier_scores.mean() if not comp_tier_scores.empty else 1
            conf_int = int(round(avg_score))
            conf_int = max(1, min(conf_int, 4))
        else:
            conf_int = 1
        st.markdown(f"<b>Confidence:</b> {conf_int}/4", unsafe_allow_html=True)

    st.markdown(f"<b>All ARVs from runs:</b> {', '.join([f'${arv:,.0f}' for arv in st.session_state.all_arvs])}")

    st.subheader("Detailed Summary Report")
    
    # CSS injection for horizontal scrolling on code blocks
    st.markdown("""
    <style>
    div[data-testid="stCodeBlock"] > div {
        overflow-x: auto !important;
        white-space: pre !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.code(st.session_state.report_str, language="text")

    st.subheader("Combined Comparables Data (All Runs)")
    df = st.session_state.adjusted_comps
    if not df.empty and 'run' in df.columns:
        cols = ['run'] + [col for col in df.columns if col != 'run']
        st.dataframe(df[cols])
    else:
        st.dataframe(df)

else:
    st.info("Please enter the subject property details in the sidebar and click 'Run Underwriting Analysis' to begin.")
    if not st.session_state.adjusted_comps.empty:
        df = st.session_state.adjusted_comps
        if not df.empty and 'run' in df.columns:
            cols = ['run'] + [col for col in df.columns if col != 'run' and col != 'id']
            st.dataframe(df[cols])
        else:
            st.dataframe(df[[col for col in df.columns if col != 'id']])