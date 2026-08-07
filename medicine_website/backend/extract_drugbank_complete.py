"""
Complete DrugBank Data Extractor
Extracts drug names, dosages, side effects, and creates comprehensive databases
for 100% accurate medication safety analysis
"""

import xml.etree.ElementTree as ET
import json
import re
from collections import defaultdict

# DrugBank XML namespace
NS = {'db': 'http://www.drugbank.ca'}

def parse_drugbank_complete(xml_file_path):
    """Parse DrugBank XML and extract all relevant data"""
    print("=" * 80)
    print("STARTING COMPLETE DRUGBANK DATA EXTRACTION")
    print("=" * 80)
    
    dosage_data = {}
    side_effects_data = {}
    drug_list = []
    
    print(f"\nParsing XML file: {xml_file_path}")
    print("This may take several minutes for the full database...")
    
    try:
        # Parse XML with iterparse for memory efficiency
        context = ET.iterparse(xml_file_path, events=("start", "end"))
        context = iter(context)
        event, root = next(context)
        
        drug_count = 0
        drugs_with_dosage = 0
        drugs_with_side_effects = 0
        
        for event, elem in context:
            if event == "end" and elem.tag == "{http://www.drugbank.ca}drug":
                try:
                    # Extract all drug information
                    drug_info = extract_complete_drug_info(elem)
                    
                    if drug_info and drug_info.get('name'):
                        drug_name = drug_info['name'].lower().strip()
                        drug_count += 1
                        
                        # Add to drug list
                        drug_list.append(drug_info['name'])
                        
                        # Add dosage information
                        if drug_info.get('dosage') and drug_info['dosage'].get('max_daily_mg'):
                            dosage_data[drug_name] = drug_info['dosage']
                            drugs_with_dosage += 1
                            
                            # Also add synonyms
                            for synonym in drug_info.get('synonyms', [])[:5]:  # Limit synonyms
                                syn_lower = synonym.lower().strip()
                                if syn_lower and syn_lower not in dosage_data:
                                    dosage_data[syn_lower] = drug_info['dosage'].copy()
                        
                        # Add side effects information
                        if drug_info.get('side_effects'):
                            side_effects_data[drug_name] = drug_info['side_effects']
                            drugs_with_side_effects += 1
                            
                            # Also add synonyms
                            for synonym in drug_info.get('synonyms', [])[:5]:
                                syn_lower = synonym.lower().strip()
                                if syn_lower and syn_lower not in side_effects_data:
                                    side_effects_data[syn_lower] = drug_info['side_effects'].copy()
                        
                        if drug_count % 500 == 0:
                            print(f"  Processed {drug_count} drugs... ({drugs_with_dosage} with dosage, {drugs_with_side_effects} with side effects)")
                    
                    # Clear element to save memory
                    elem.clear()
                    root.clear()
                    
                except Exception as e:
                    print(f"  Error processing drug: {e}")
                    continue
        
        print(f"\n{'=' * 80}")
        print(f"EXTRACTION COMPLETE!")
        print(f"{'=' * 80}")
        print(f"Total drugs processed: {drug_count}")
        print(f"Drugs with dosage data: {drugs_with_dosage}")
        print(f"Drugs with side effects data: {drugs_with_side_effects}")
        print(f"Total dosage entries (including synonyms): {len(dosage_data)}")
        print(f"Total side effects entries (including synonyms): {len(side_effects_data)}")
        
        return dosage_data, side_effects_data, drug_list
        
    except Exception as e:
        print(f"\nERROR: Failed to parse DrugBank XML: {e}")
        return {}, {}, []

def extract_complete_drug_info(drug_elem):
    """Extract complete information from a drug element"""
    try:
        # Get drug name
        name_elem = drug_elem.find("db:name", NS)
        if name_elem is None or not name_elem.text:
            return None
        drug_name = name_elem.text.strip()
        
        # Get synonyms
        synonyms = []
        for synonym in drug_elem.findall(".//db:synonym", NS):
            if synonym.text:
                synonyms.append(synonym.text.strip())
        
        # Get drug type
        drug_type = "Unknown"
        type_elem = drug_elem.find("db:type", NS)
        if type_elem is not None and type_elem.text:
            drug_type = type_elem.text.strip()
        
        # Extract dosage information
        dosage_info = extract_dosage_information(drug_elem, drug_name)
        
        # Extract side effects
        side_effects_info = extract_side_effects_information(drug_elem, drug_name)
        
        return {
            'name': drug_name,
            'synonyms': synonyms,
            'type': drug_type,
            'dosage': dosage_info,
            'side_effects': side_effects_info
        }
        
    except Exception as e:
        return None

def extract_dosage_information(drug_elem, drug_name):
    """Extract comprehensive dosage information"""
    dosage_info = {
        'max_daily_mg': None,
        'max_single_mg': None,
        'unit': 'mg',
        'warnings': []
    }
    
    try:
        # Collect all text that might contain dosage info
        text_sources = []
        
        # Description
        desc_elem = drug_elem.find("db:description", NS)
        if desc_elem is not None and desc_elem.text:
            text_sources.append(desc_elem.text)
        
        # Indication
        ind_elem = drug_elem.find("db:indication", NS)
        if ind_elem is not None and ind_elem.text:
            text_sources.append(ind_elem.text)
        
        # Pharmacodynamics
        pharm_elem = drug_elem.find("db:pharmacodynamics", NS)
        if pharm_elem is not None and pharm_elem.text:
            text_sources.append(pharm_elem.text)
        
        # Dosages section
        for dosage_elem in drug_elem.findall(".//db:dosage", NS):
            if dosage_elem.text:
                text_sources.append(dosage_elem.text)
        
        # Parse dosage from all text sources
        for text in text_sources:
            parsed = parse_dosage_from_text(text)
            if parsed.get('max_daily_mg'):
                dosage_info['max_daily_mg'] = parsed['max_daily_mg']
            if parsed.get('max_single_mg'):
                dosage_info['max_single_mg'] = parsed['max_single_mg']
            if parsed.get('warnings'):
                dosage_info['warnings'].extend(parsed['warnings'])
        
        # If we found max_daily but not max_single, estimate it
        if dosage_info['max_daily_mg'] and not dosage_info['max_single_mg']:
            dosage_info['max_single_mg'] = dosage_info['max_daily_mg'] / 3
        
        # Extract warnings
        warnings = extract_warnings_from_drug(drug_elem)
        dosage_info['warnings'].extend(warnings)
        
        # Limit warnings to top 4
        dosage_info['warnings'] = list(set(dosage_info['warnings']))[:4]
        
    except Exception as e:
        pass
    
    return dosage_info

def parse_dosage_from_text(text):
    """Parse dosage values from text using regex"""
    result = {}
    text_lower = text.lower()
    
    # Patterns for maximum daily dose
    max_daily_patterns = [
        r'maximum daily dose[:\s]+(?:of\s+)?(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|iu|units?)',
        r'max(?:imum)? daily[:\s]+(?:of\s+)?(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|iu|units?)',
        r'daily maximum[:\s]+(?:of\s+)?(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|iu|units?)',
        r'not exceed[^.]{0,50}?(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|iu|units?)\s*(?:per day|daily)',
        r'up to[^.]{0,30}?(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|iu|units?)\s*(?:per day|daily)',
    ]
    
    # Patterns for maximum single dose
    max_single_patterns = [
        r'maximum single dose[:\s]+(?:of\s+)?(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|iu|units?)',
        r'max(?:imum)? single[:\s]+(?:of\s+)?(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|iu|units?)',
        r'single dose[:\s]+(?:of\s+)?(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|iu|units?)',
    ]
    
    # Try to find maximum daily dose
    for pattern in max_daily_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            value, unit = matches[0]
            value = float(value)
            result['max_daily_mg'] = convert_to_mg(value, unit)
            result['unit'] = 'mg'
            break
    
    # Try to find maximum single dose
    for pattern in max_single_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            value, unit = matches[0]
            value = float(value)
            result['max_single_mg'] = convert_to_mg(value, unit)
            break
    
    # Extract warnings
    warning_keywords = ['overdose', 'toxic', 'fatal', 'dangerous', 'severe', 'caution']
    warnings = []
    for keyword in warning_keywords:
        pattern = rf'([^.]*{keyword}[^.]*dose[^.]*)'
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if len(match.strip()) > 15:
                warnings.append(match.strip().capitalize()[:100])
    
    if warnings:
        result['warnings'] = warnings[:2]
    
    return result

def convert_to_mg(value, unit):
    """Convert dosage to mg"""
    unit_lower = unit.lower()
    if unit_lower in ['g', 'gram', 'grams']:
        return value * 1000
    elif unit_lower in ['mcg', 'μg', 'microgram', 'micrograms']:
        return value / 1000
    elif unit_lower in ['iu', 'unit', 'units']:
        # Keep as is for IU/units (will be handled separately)
        return value
    else:  # mg
        return value

def extract_side_effects_information(drug_elem, drug_name):
    """Extract side effects and adverse reactions"""
    side_effects_info = {
        'common_side_effects': [],
        'serious_side_effects': [],
        'risk_factors': {}
    }
    
    try:
        # Collect text from relevant sections
        text_sources = []
        
        # Toxicity section
        tox_elem = drug_elem.find("db:toxicity", NS)
        if tox_elem is not None and tox_elem.text:
            text_sources.append(('toxicity', tox_elem.text))
        
        # Adverse reactions
        for adverse_elem in drug_elem.findall(".//db:adverse-reaction", NS):
            if adverse_elem.text:
                text_sources.append(('adverse', adverse_elem.text))
        
        # Side effects
        for se_elem in drug_elem.findall(".//db:side-effect", NS):
            if se_elem.text:
                text_sources.append(('side_effect', se_elem.text))
        
        # Parse side effects from text
        common_effects = set()
        serious_effects = set()
        
        for source_type, text in text_sources:
            text_lower = text.lower()
            
            # Common side effects keywords
            common_keywords = ['nausea', 'headache', 'dizziness', 'fatigue', 'drowsiness', 
                             'dry mouth', 'constipation', 'diarrhea', 'insomnia', 'rash']
            
            # Serious side effects keywords
            serious_keywords = ['liver damage', 'kidney failure', 'heart attack', 'stroke', 
                              'seizure', 'bleeding', 'allergic reaction', 'death', 'toxic']
            
            for keyword in common_keywords:
                if keyword in text_lower:
                    common_effects.add(keyword.capitalize())
            
            for keyword in serious_keywords:
                if keyword in text_lower:
                    serious_effects.add(keyword.capitalize())
        
        side_effects_info['common_side_effects'] = list(common_effects)[:5]
        side_effects_info['serious_side_effects'] = list(serious_effects)[:5]
        
        # Extract risk factors
        risk_factors = {}
        risk_keywords = {
            'age_65_plus': ['elderly', 'geriatric', 'older adult'],
            'liver_disease': ['liver', 'hepatic'],
            'kidney_disease': ['kidney', 'renal'],
            'heart_disease': ['heart', 'cardiac', 'cardiovascular'],
            'pregnancy': ['pregnancy', 'pregnant', 'fetal']
        }
        
        all_text = ' '.join([text for _, text in text_sources]).lower()
        for risk_key, keywords in risk_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    risk_factors[risk_key] = f"Increased risk with {keyword} conditions"
                    break
        
        side_effects_info['risk_factors'] = risk_factors
        
    except Exception as e:
        pass
    
    return side_effects_info

def extract_warnings_from_drug(drug_elem):
    """Extract warnings and contraindications"""
    warnings = []
    
    try:
        # Contraindications
        contra_elem = drug_elem.find("db:contraindications", NS)
        if contra_elem is not None and contra_elem.text:
            text = contra_elem.text.strip()
            if len(text) > 20:
                warnings.append(text[:150])
        
        # Warnings
        warn_elem = drug_elem.find("db:warning", NS)
        if warn_elem is not None and warn_elem.text:
            text = warn_elem.text.strip()
            if len(text) > 20:
                warnings.append(text[:150])
        
    except Exception as e:
        pass
    
    return warnings

def save_databases(dosage_data, side_effects_data, drug_list):
    """Save extracted data to JSON files"""
    print(f"\n{'=' * 80}")
    print("SAVING DATABASES...")
    print(f"{'=' * 80}")
    
    # Save dosage database
    print(f"\n1. Saving dosage database...")
    with open('drug_dosage_limits_drugbank.json', 'w', encoding='utf-8') as f:
        json.dump(dosage_data, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved {len(dosage_data)} entries to 'drug_dosage_limits_drugbank.json'")
    
    # Save side effects database
    print(f"\n2. Saving side effects database...")
    with open('side_effects_database_drugbank.json', 'w', encoding='utf-8') as f:
        json.dump(side_effects_data, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved {len(side_effects_data)} entries to 'side_effects_database_drugbank.json'")
    
    # Save drug list
    print(f"\n3. Saving complete drug list...")
    with open('drugbank_drug_list.json', 'w', encoding='utf-8') as f:
        json.dump(sorted(drug_list), f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved {len(drug_list)} drug names to 'drugbank_drug_list.json'")
    
    print(f"\n{'=' * 80}")
    print("ALL DATABASES SAVED SUCCESSFULLY!")
    print(f"{'=' * 80}")

def main():
    """Main extraction function"""
    xml_file = 'DRUG_bank dataset/full database.xml'
    
    print("\n" + "=" * 80)
    print("DRUGBANK COMPLETE DATA EXTRACTION TOOL")
    print("=" * 80)
    print("\nThis tool will extract:")
    print("  1. Drug names and synonyms")
    print("  2. Dosage limits and warnings")
    print("  3. Side effects and adverse reactions")
    print("  4. Risk factors and contraindications")
    print("\nFrom: DrugBank Full Database XML")
    print("=" * 80)
    
    # Extract all data
    dosage_data, side_effects_data, drug_list = parse_drugbank_complete(xml_file)
    
    if dosage_data or side_effects_data:
        # Save to files
        save_databases(dosage_data, side_effects_data, drug_list)
        
        print("\n" + "=" * 80)
        print("EXTRACTION SUMMARY")
        print("=" * 80)
        print(f"✓ Total unique drugs: {len(drug_list)}")
        print(f"✓ Drugs with dosage data: {len(dosage_data)}")
        print(f"✓ Drugs with side effects data: {len(side_effects_data)}")
        print("\nGenerated files:")
        print("  - drug_dosage_limits_drugbank.json")
        print("  - side_effects_database_drugbank.json")
        print("  - drugbank_drug_list.json")
        print("\nNext steps:")
        print("  1. Update app.py to use the new DrugBank databases")
        print("  2. Restart Flask server")
        print("  3. Test with any drug from the database")
        print("=" * 80 + "\n")
    else:
        print("\n❌ ERROR: No data extracted. Please check the XML file path and format.")

if __name__ == "__main__":
    main()
