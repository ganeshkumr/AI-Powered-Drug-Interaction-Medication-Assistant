import xml.etree.ElementTree as ET
import json
import re
from collections import defaultdict
import time

def parse_drugbank_efficient(xml_file_path, max_drugs=1000):
    """Efficiently parse DrugBank XML file and extract dosage information"""
    print(f"Starting to parse DrugBank XML file (max {max_drugs} drugs)...")
    
    dosage_data = {}
    drug_count = 0
    
    # Use iterparse for memory-efficient parsing
    context = ET.iterparse(xml_file_path, events=("start", "end"))
    context = iter(context)
    event, root = next(context)
    
    for event, elem in context:
        if event == "end" and elem.tag == "drug":
            try:
                drug_info = extract_drug_info_efficient(elem)
                if drug_info and drug_info.get('name'):
                    drug_name = drug_info['name'].lower().strip()
                    dosage_data[drug_name] = drug_info
                    drug_count += 1
                    
                    if drug_count % 100 == 0:
                        print(f"Processed {drug_count} drugs...")
                    
                    if drug_count >= max_drugs:
                        print(f"Reached maximum limit of {max_drugs} drugs")
                        break
                
                # Clear element to save memory
                elem.clear()
                root.clear()
                
            except Exception as e:
                print(f"Error processing drug: {e}")
                continue
    
    print(f"Successfully parsed {drug_count} drugs from DrugBank")
    return dosage_data

def extract_drug_info_efficient(drug_elem):
    """Extract relevant information from a drug element efficiently"""
    try:
        # Get drug name
        name_elem = drug_elem.find(".//{http://www.drugbank.ca}name")
        if name_elem is None or not name_elem.text:
            return None
        drug_name = name_elem.text.strip()
        
        # Get synonyms for better matching
        synonyms = []
        for synonym in drug_elem.findall(".//{http://www.drugbank.ca}synonym"):
            if synonym.text:
                synonyms.append(synonym.text.strip().lower())
        
        # Get dosage information from description only (most efficient)
        dosage_info = extract_dosage_from_description(drug_elem)
        
        # Get drug type
        drug_type = "Unknown"
        type_elem = drug_elem.find(".//{http://www.drugbank.ca}type")
        if type_elem is not None and type_elem.text:
            drug_type = type_elem.text.strip()
        
        # Get basic warnings
        warnings = extract_basic_warnings(drug_elem)
        
        return {
            'name': drug_name,
            'synonyms': synonyms[:5],  # Limit synonyms
            'type': drug_type,
            'dosage': dosage_info,
            'warnings': warnings[:3]  # Limit warnings
        }
        
    except Exception as e:
        print(f"Error extracting drug info: {e}")
        return None

def extract_dosage_from_description(drug_elem):
    """Extract dosage information from description only"""
    dosage_info = {
        'max_daily_mg': None,
        'max_single_mg': None,
        'unit': 'mg',
        'warnings': []
    }
    
    try:
        # Only check description for efficiency
        description_elem = drug_elem.find(".//{http://www.drugbank.ca}description")
        if description_elem is not None and description_elem.text:
            description = description_elem.text
            dosage_info.update(parse_dosage_from_text(description))
        
    except Exception as e:
        print(f"Error extracting dosage info: {e}")
    
    return dosage_info

def parse_dosage_from_text(text):
    """Parse dosage information from text using regex patterns"""
    dosage_info = {}
    text_lower = text.lower()
    
    # Common dosage patterns
    patterns = {
        'max_daily': [
            r'maximum daily dose[:\s]*(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
            r'daily maximum[:\s]*(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
            r'max daily[:\s]*(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
            r'not exceed[^.]*?(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
            r'daily dose[:\s]*(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
        ],
        'max_single': [
            r'maximum single dose[:\s]*(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
            r'single maximum[:\s]*(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
            r'max single[:\s]*(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
        ]
    }
    
    for category, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, text_lower)
            if matches:
                value, unit = matches[0]
                value = float(value)
                
                # Convert to mg for consistency
                if unit in ['g', 'gram']:
                    value = value * 1000
                elif unit in ['mcg', 'μg', 'microgram']:
                    value = value / 1000
                
                if category == 'max_daily':
                    dosage_info['max_daily_mg'] = value
                elif category == 'max_single':
                    dosage_info['max_single_mg'] = value
                
                break
    
    # Extract basic warnings
    warning_patterns = [
        r'warning[^.]*?([^.]*?overdose[^.]*)',
        r'caution[^.]*?([^.]*?dose[^.]*)',
        r'contraindication[^.]*?([^.]*?dose[^.]*)',
    ]
    
    warnings = []
    for pattern in warning_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if len(match.strip()) > 10:
                warnings.append(match.strip().capitalize())
    
    if warnings:
        dosage_info['warnings'] = warnings[:2]  # Limit warnings
    
    return dosage_info

def extract_basic_warnings(drug_elem):
    """Extract basic warnings efficiently"""
    warnings = []
    
    try:
        # Only check contraindications for efficiency
        contraindication_elem = drug_elem.find(".//{http://www.drugbank.ca}contraindication")
        if contraindication_elem is not None and contraindication_elem.text:
            warnings.append(contraindication_elem.text.strip()[:100])  # Limit length
        
    except Exception as e:
        print(f"Error extracting warnings: {e}")
    
    return warnings

def create_enhanced_dosage_database():
    """Create enhanced dosage database from DrugBank data"""
    print("Creating enhanced dosage database from DrugBank...")
    
    # Parse DrugBank XML with limit
    drugbank_data = parse_drugbank_efficient('DRUG_bank dataset/full database.xml', max_drugs=2000)
    
    # Load existing dosage limits
    try:
        with open('drug_dosage_limits.json', 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = {}
    
    # Merge data
    enhanced_data = existing_data.copy()
    new_entries = 0
    
    for drug_name, drug_info in drugbank_data.items():
        if drug_info.get('dosage') and drug_info['dosage'].get('max_daily_mg'):
            # Convert to our format
            dosage_info = drug_info['dosage']
            enhanced_data[drug_name] = {
                'max_daily_mg': dosage_info.get('max_daily_mg'),
                'max_single_mg': dosage_info.get('max_single_mg', dosage_info.get('max_daily_mg', 0) / 3),
                'unit': 'mg',
                'warnings': dosage_info.get('warnings', []) + drug_info.get('warnings', [])
            }
            new_entries += 1
            
            # Add synonyms
            for synonym in drug_info.get('synonyms', []):
                if synonym not in enhanced_data:
                    enhanced_data[synonym] = enhanced_data[drug_name].copy()
                    new_entries += 1
    
    # Save enhanced database
    with open('drug_dosage_limits_enhanced.json', 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    print(f"Enhanced dosage database created with {len(enhanced_data)} total entries")
    print(f"Added {new_entries} new entries from DrugBank")
    return enhanced_data

if __name__ == "__main__":
    start_time = time.time()
    create_enhanced_dosage_database()
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")

