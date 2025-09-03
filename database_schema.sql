-- Supabase Database Schema for AI Underwriting Agent
-- Execute this in your Supabase SQL Editor

-- Main properties table
CREATE TABLE properties (
  id BIGSERIAL PRIMARY KEY,
  zpid BIGINT UNIQUE,
  latitude DECIMAL(10,8),
  longitude DECIMAL(11,8),
  street_address TEXT,
  city TEXT,
  state TEXT,
  zipcode TEXT,
  bathrooms DECIMAL(3,1),
  bedrooms INTEGER,
  living_area INTEGER,
  year_built INTEGER,
  lot_size DECIMAL(12,3),
  lot_size_unit TEXT,
  property_type TEXT,
  price INTEGER,
  last_sold_date DATE,
  listing_status TEXT,
  tax_assessed_value INTEGER,
  tax_assessment_year INTEGER,
  hdp_url TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Property images table
CREATE TABLE property_images (
  id BIGSERIAL PRIMARY KEY,
  property_id BIGINT REFERENCES properties(id) ON DELETE CASCADE,
  image_url TEXT NOT NULL,
  image_type TEXT CHECK (image_type IN ('main', 'streetview', 'satellite', 'all_photos')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_properties_location ON properties USING GIST (
  point(longitude, latitude)
);
CREATE INDEX idx_properties_sold_date ON properties (last_sold_date);
CREATE INDEX idx_properties_city_state ON properties (city, state);
CREATE INDEX idx_properties_property_type ON properties (property_type);
CREATE INDEX idx_properties_price ON properties (price);
CREATE INDEX idx_properties_bedrooms_bathrooms ON properties (bedrooms, bathrooms);
CREATE INDEX idx_properties_living_area ON properties (living_area);

-- Enable Row Level Security (RLS)
ALTER TABLE properties ENABLE ROW LEVEL SECURITY;
ALTER TABLE property_images ENABLE ROW LEVEL SECURITY;

-- Create policies for public read access
CREATE POLICY "Public read access for properties" ON properties
  FOR SELECT USING (true);

CREATE POLICY "Public read access for property_images" ON property_images
  FOR SELECT USING (true);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_properties_updated_at
  BEFORE UPDATE ON properties
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Example query to test geographic search (for reference)
-- SELECT * FROM properties 
-- WHERE point(longitude, latitude) <@> point(-122.3321, 47.6062) < 0.1
-- ORDER BY point(longitude, latitude) <@> point(-122.3321, 47.6062)
-- LIMIT 10;