"""
Ultra-fast image migration that only queries zpids that have images
"""

import pandas as pd
import ast
import requests
import time
import argparse
from typing import List, Dict

def migrate_images_ultra_fast(csv_path='washington_cleaned.csv', dry_run=True, chunk_size=1000, debug=False):
    """Ultra-fast migration that queries zpids on-demand"""
    
    print(f"Ultra-fast image migration: {'DRY RUN' if dry_run else 'LIVE'}")
    
    from database_client_rest import get_rest_database_client
    client = get_rest_database_client()
    
    stats = {
        'total_processed': 0,
        'properties_with_images': 0,
        'zpids_found': 0,
        'zpids_not_found': 0,
        'images_inserted': 0,
        'insertion_errors': 0
    }
    
    start_time = time.time()
    all_image_records = []
    
    # Process CSV in chunks
    chunk_num = 0
    for df_chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        chunk_num += 1
        chunk_start = (chunk_num - 1) * chunk_size + 1
        
        # Filter chunk for properties with images
        has_images = (
            df_chunk['media.allPropertyPhotos.highResolution'].notna() & 
            (df_chunk['media.allPropertyPhotos.highResolution'] != 'False') &
            (df_chunk['media.allPropertyPhotos.highResolution'] != False)
        )
        chunk_with_images = df_chunk[has_images]
        
        stats['total_processed'] += len(df_chunk)
        stats['properties_with_images'] += len(chunk_with_images)
        
        if len(chunk_with_images) == 0:
            print(f"Chunk {chunk_num}: {len(df_chunk)} rows, 0 with images")
            continue
        
        print(f"Chunk {chunk_num} (rows {chunk_start:,}+): {len(df_chunk)} rows, {len(chunk_with_images)} with images")
        
        # Get zpids that have images
        zpids_with_images = chunk_with_images['zpid'].tolist()
        
        # Single batch query for all zpids in this chunk
        zpids_str = ','.join(map(str, zpids_with_images))
        query_url = f'{client.url}/rest/v1/properties?select=id,zpid&zpid=in.({zpids_str})'
        
        try:
            response = requests.get(query_url, headers=client.headers, timeout=60)
            if response.status_code == 200:
                found_props = response.json()
                zpid_to_id = {prop['zpid']: prop['id'] for prop in found_props}
                
                stats['zpids_found'] += len(found_props)
                stats['zpids_not_found'] += len(zpids_with_images) - len(found_props)
                
                # Process images for found properties
                for _, row in chunk_with_images.iterrows():
                    zpid = row['zpid']
                    property_id = zpid_to_id.get(zpid)
                    
                    if property_id:
                        # Parse images
                        try:
                            images = ast.literal_eval(str(row['media.allPropertyPhotos.highResolution']))
                            if isinstance(images, list):
                                for img_url in images:
                                    if isinstance(img_url, str) and img_url.startswith('http'):
                                        all_image_records.append({
                                            'property_id': property_id,
                                            'image_url': img_url,
                                            'image_type': 'all_photos'
                                        })
                        except:
                            continue
                
                print(f"  Found {len(found_props)} properties, prepared {len(all_image_records)} total images")
                
            else:
                print(f"  ERROR: Query failed {response.status_code}")
                
        except Exception as e:
            print(f"  ERROR: Query exception {e}")
        
        # Insert in large batches for speed
        if len(all_image_records) >= 2000 or chunk_num % 50 == 0:
            if all_image_records:
                insert_count = len(all_image_records)
                if dry_run:
                    print(f"  [DRY RUN] Would insert {insert_count} images")
                    stats['images_inserted'] += insert_count
                else:
                    # Live insertion
                    print(f"  [LIVE] Inserting {insert_count} images...")
                    success = insert_images_batch(client, all_image_records)
                    if success:
                        stats['images_inserted'] += insert_count
                        print(f"    SUCCESS: Inserted {insert_count} images")
                    else:
                        stats['insertion_errors'] += insert_count
                        print(f"    ERROR: Failed to insert {insert_count} images")
                
                all_image_records = []  # Clear batch
        
        # Progress update
        if chunk_num % 10 == 0:
            elapsed = time.time() - start_time
            rate = stats['total_processed'] / elapsed
            print(f"  Progress: {stats['total_processed']:,} processed ({rate:.0f} rows/sec), {stats['images_inserted']:,} images ready")
    
    # Insert final batch
    if all_image_records:
        insert_count = len(all_image_records)
        if dry_run:
            print(f"[DRY RUN] Final batch: would insert {insert_count} images")
            stats['images_inserted'] += insert_count
        else:
            print(f"[LIVE] Final batch: inserting {insert_count} images...")
            success = insert_images_batch(client, all_image_records)
            if success:
                stats['images_inserted'] += insert_count
            else:
                stats['insertion_errors'] += insert_count
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n=== ULTRA-FAST MIGRATION SUMMARY ===")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Processing rate: {stats['total_processed'] / total_time:.0f} rows/sec")
    print(f"Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    return stats['insertion_errors'] == 0

def insert_images_batch(client, image_records):
    """Insert batch of images"""
    try:
        url = f"{client.url}/rest/v1/property_images"
        response = requests.post(
            url,
            headers={**client.headers, 'Prefer': 'return=minimal'},
            json=image_records,
            timeout=120
        )
        return response.status_code in [200, 201]
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description='Ultra-fast image migration')
    parser.add_argument('--csv-path', default='washington_cleaned.csv')
    parser.add_argument('--live', action='store_true', help='Live migration')
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test-small', action='store_true', help='Test on first 5000 rows')
    
    args = parser.parse_args()
    dry_run = not args.live
    
    # Limit for testing
    if args.test_small:
        print("TEST MODE: First 5000 rows only")
        chunk_size = min(args.chunk_size, 1000)
        # Read only first 5000 rows by creating temp file
        import pandas as pd
        df_test = pd.read_csv(args.csv_path, nrows=5000, low_memory=False)
        test_path = 'temp_test_5k.csv'
        df_test.to_csv(test_path, index=False)
        result = migrate_images_ultra_fast(test_path, dry_run, chunk_size, args.debug)
        import os
        os.remove(test_path)
        return result
    else:
        return migrate_images_ultra_fast(args.csv_path, dry_run, args.chunk_size, args.debug)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)