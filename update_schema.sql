-- Update schema to allow decimal bedrooms
-- Execute this in your Supabase SQL Editor

-- Change bedrooms from INTEGER to DECIMAL to allow fractional bedrooms
ALTER TABLE properties ALTER COLUMN bedrooms TYPE DECIMAL(3,1);

-- Verify the change
SELECT column_name, data_type, numeric_precision, numeric_scale 
FROM information_schema.columns 
WHERE table_name = 'properties' 
AND column_name IN ('bedrooms', 'bathrooms');