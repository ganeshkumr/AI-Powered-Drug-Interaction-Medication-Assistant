import xml.etree.ElementTree as ET
import json
import re
from collections import defaultdict

def parse_drugbank_xml(xml_file_path):
    """Parse DrugBank XML file and extract dosage information"""
    print("Starting to parse DrugBank XML file...")
    
    dosage_data = {}
    
    # Parse XML in chunks to handle large file
    context = ET.iterparse(xml_file_path, events=("start", "end"))
    context = iter(context)
    event, root = next(context)
    
    drug_count = 0
    
    for event, elem in context:
        if event == "end" and elem.tag == "drug":
            try:
                drug_info = extract_drug_info(elem)
                if drug_info:
                    drug_name = drug_info.get('name', '').lower().strip()
                    if drug_name:
                        dosage_data[drug_name] = drug_info
                        drug_count += 1
                        if drug_count % 1000 == 0:
                            print(f"Processed {drug_count} drugs...")
                
                # Clear element to save memory
                elem.clear()
                root.clear()
                
            except Exception as e:
                print(f"Error processing drug: {e}")
                continue
    
    print(f"Successfully parsed {drug_count} drugs from DrugBank")
    return dosage_data

def extract_drug_info(drug_elem):
    """Extract relevant information from a drug element"""
    try:
        # Get drug name
        name_elem = drug_elem.find(".//{http://www.drugbank.ca}name")
        if name_elem is None:
            return None
        drug_name = name_elem.text.strip()
        
        # Get synonyms for better matching
        synonyms = []
        for synonym in drug_elem.findall(".//{http://www.drugbank.ca}synonym"):
            if synonym.text:
                synonyms.append(synonym.text.strip().lower())
        
        # Get dosage information
        dosage_info = extract_dosage_info(drug_elem)
        
        # Get drug type and categories
        drug_type = "Unknown"
        type_elem = drug_elem.find(".//{http://www.drugbank.ca}type")
        if type_elem is not None:
            drug_type = type_elem.text.strip()
        
        # Get warnings and contraindications
        warnings = extract_warnings(drug_elem)
        
        return {
            'name': drug_name,
            'synonyms': synonyms,
            'type': drug_type,
            'dosage': dosage_info,
            'warnings': warnings
        }
        
    except Exception as e:
        print(f"Error extracting drug info: {e}")
        return None

def extract_dosage_info(drug_elem):
    """Extract dosage information from drug element"""
    dosage_info = {
        'max_daily_mg': None,
        'max_single_mg': None,
        'unit': 'mg',
        'warnings': []
    }
    
    try:
        # Look for dosage information in various sections
        # Check for dosage information in the description
        description_elem = drug_elem.find(".//{http://www.drugbank.ca}description")
        if description_elem is not None:
            description = description_elem.text or ""
            dosage_info.update(parse_dosage_from_text(description))
        
        # Check for dosage in indications
        indication_elem = drug_elem.find(".//{http://www.drugbank.ca}indication")
        if indication_elem is not None:
            indication = indication_elem.text or ""
            dosage_info.update(parse_dosage_from_text(indication))
        
        # Check for dosage in pharmacodynamics
        pharm_elem = drug_elem.find(".//{http://www.drugbank.ca}pharmacodynamics")
        if pharm_elem is not None:
            pharm_text = pharm_elem.text or ""
            dosage_info.update(parse_dosage_from_text(pharm_text))
        
        # Look for specific dosage elements
        dosage_elem = drug_elem.find(".//{http://www.drugbank.ca}dosage")
        if dosage_elem is not None:
            dosage_text = dosage_elem.text or ""
            dosage_info.update(parse_dosage_from_text(dosage_text))
        
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
        ],
        'max_single': [
            r'maximum single dose[:\s]*(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
            r'single maximum[:\s]*(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
            r'max single[:\s]*(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
        ],
        'typical_daily': [
            r'usual daily dose[:\s]*(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
            r'daily dose[:\s]*(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
            r'standard dose[:\s]*(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg)',
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
                elif category == 'typical_daily' and 'max_daily_mg' not in dosage_info:
                    # Use typical as max if no explicit max found
                    dosage_info['max_daily_mg'] = value * 1.5  # Add 50% safety margin
                
                break
    
    # Extract warnings
    warning_patterns = [
        r'warning[^.]*?([^.]*?overdose[^.]*)',
        r'caution[^.]*?([^.]*?dose[^.]*)',
        r'contraindication[^.]*?([^.]*?dose[^.]*)',
        r'not exceed[^.]*?([^.]*?mg[^.]*)',
    ]
    
    warnings = []
    for pattern in warning_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if len(match.strip()) > 10:  # Only meaningful warnings
                warnings.append(match.strip().capitalize())
    
    if warnings:
        dosage_info['warnings'] = warnings[:3]  # Limit to 3 warnings
    
    return dosage_info

def extract_warnings(drug_elem):
    """Extract warnings and contraindications"""
    warnings = []
    
    try:
        # Check for contraindications
        contraindication_elem = drug_elem.find(".//{http://www.drugbank.ca}contraindication")
        if contraindication_elem is not None and contraindication_elem.text:
            warnings.append(contraindication_elem.text.strip())
        
        # Check for warnings
        warning_elem = drug_elem.find(".//{http://www.drugbank.ca}warning")
        if warning_elem is not None and warning_elem.text:
            warnings.append(warning_elem.text.strip())
        
        # Check for adverse reactions
        adverse_elem = drug_elem.find(".//{http://www.drugbank.ca}adverse-reaction")
        if adverse_elem is not None and adverse_elem.text:
            warnings.append(f"Adverse reactions: {adverse_elem.text.strip()}")
        
    except Exception as e:
        print(f"Error extracting warnings: {e}")
    
    return warnings[:3]  # Limit to 3 warnings

def create_enhanced_dosage_database():
    """Create enhanced dosage database from DrugBank data"""
    print("Creating enhanced dosage database from DrugBank...")
    
    # Parse DrugBank XML
    drugbank_data = parse_drugbank_xml('DRUG_bank dataset/full database.xml')
    
    # Load existing dosage limits
    try:
        with open('drug_dosage_limits.json', 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = {}
    
    # Merge data
    enhanced_data = existing_data.copy()
    
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
            
            # Add synonyms
            for synonym in drug_info.get('synonyms', []):
                if synonym not in enhanced_data:
                    enhanced_data[synonym] = enhanced_data[drug_name].copy()
    
    # Save enhanced database
    with open('drug_dosage_limits_enhanced.json', 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    print(f"Enhanced dosage database created with {len(enhanced_data)} entries")
    return enhanced_data

if __name__ == "__main__":
    create_enhanced_dosage_database()
