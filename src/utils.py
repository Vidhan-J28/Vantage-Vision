#saving the metadata
import json
from pathlib import Path
import pandas as pd

def ensure_analyzed_dir(raw_path):
    raw_path = Path(raw_path)
    analyzed_path = raw_path / "Analyzed" / raw_path.name
    analyzed_path.mkdir(parents=True , exist_ok=True)
    return analyzed_path

    
def save(data, raw_path):
    analyzed = ensure_analyzed_dir(raw_path)
    
    output_path = analyzed / "ALL_DATA.json"
    
    with open(output_path , 'w') as f:
        json.dump(data , f)
        
    return output_path

def load_data(metadata_path):
    m_path = Path(metadata_path)
    
    if not m_path.exists():
        analyzed_path = m_path.parent.parent / "Analyzed" / m_path.name / "ALL_DATA.json"
        if analyzed_path.exists():
            m_path=analyzed_path
        else:
            raise FileNotFoundError("No data found at the given location")
        
    with open(m_path, 'r') as f:
        return json.load(f)
        
def get_class_freq(data):
    unique_labels = set()
    freq_labels={}
    
    for item in data:
        for clss in item['detections']:
            unique_labels.add(clss['class'])
            
            if clss['class'] not in freq_labels:
                freq_labels[clss['class']] = set()
                
            freq_labels[clss['class']].add(clss['count'])
            
    unique_labels=sorted(unique_labels)
    
    for clss in freq_labels:
        freq_labels[clss]=sorted(freq_labels[clss])
        
    return unique_labels,freq_labels
        
        
    