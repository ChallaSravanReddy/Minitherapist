import pandas as pd
import json
import os
from collections import Counter
import re

def preprocess_text(text):
    """Clean and preprocess text data."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s\'\-\.\,\!\?]', '', text)
    
    return text.strip()

def map_emotion_to_standard(emotion):
    """Map diverse emotion labels to 9 standard categories."""
    if pd.isna(emotion):
        return 'neutral'
    
    emotion = str(emotion).lower().strip()
    
    # Emotion mapping dictionary
    emotion_map = {
        # Sad variants
        'sad': 'sad',
        'sadness': 'sad',
        'sentimental': 'sad',
        'disappointed': 'sad',
        'devastated': 'sad',
        'grief': 'sad',
        'heartbroken': 'sad',
        'hopeless': 'sad',
        'depressed': 'sad',
        
        # Stressed/Anxious variants
        'stressed': 'stressed',
        'anxious': 'stressed',
        'anxiety': 'stressed',
        'nervous': 'stressed',
        'terrified': 'stressed',
        'afraid': 'stressed',
        'apprehensive': 'stressed',
        'panic': 'stressed',
        
        # Angry variants
        'angry': 'angry',
        'anger': 'angry',
        'annoyed': 'angry',
        'furious': 'angry',
        'frustrated': 'angry',
        'irritated': 'angry',
        'disgusted': 'angry',
        'jealous': 'angry',
        'bitter': 'angry',
        
        # Lonely variants
        'lonely': 'lonely',
        'alone': 'lonely',
        'isolated': 'lonely',
        'abandoned': 'lonely',
        'neglected': 'lonely',
        
        # Confused variants
        'confused': 'confused',
        'surprised': 'confused',
        'embarrassed': 'confused',
        'ashamed': 'confused',
        'guilty': 'confused',
        
        # Overwhelmed variants
        'overwhelmed': 'overwhelmed',
        'caring': 'overwhelmed',  # From empathetic dialogues
        'prepared': 'overwhelmed',
        
        # Happy variants
        'happy': 'happy',
        'joy': 'happy',
        'joyful': 'happy',
        'excited': 'happy',
        'proud': 'happy',
        'grateful': 'happy',
        'content': 'happy',
        'confident': 'happy',
        'impressed': 'happy',
        'anticipating': 'happy',
        'trusting': 'happy',
        
        # Worried variants
        'worried': 'worried',
        'fear': 'worried',
        'fearful': 'worried',
        'scared': 'worried',
        
        # Neutral variants
        'neutral': 'neutral',
        'no emotion': 'neutral',
        'non-neutral': 'neutral',  # Default fallback
    }
    
    return emotion_map.get(emotion, 'neutral')

def load_empathetic_dialogues(file_path):
    """Load and process the emotion-emotion_69k.csv file."""
    print(f"\nLoading Empathetic Dialogues from {file_path}...")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"  Loaded {len(df)} rows")
        print(f"  Columns: {df.columns.tolist()}")
        
        # The file has columns: Situation, emotion, empathetic_dialogues, labels
        data = []
        
        for idx, row in df.iterrows():
            # Use the empathetic_dialogues column as text
            text = row.get('empathetic_dialogues', row.get('Situation', ''))
            emotion = row.get('emotion', row.get('labels', 'neutral'))
            
            if pd.notna(text) and len(str(text).strip()) > 5:
                processed_text = preprocess_text(text)
                if processed_text:
                    data.append({
                        'text': processed_text,
                        'emotion': map_emotion_to_standard(emotion),
                        'source': 'empathetic_dialogues'
                    })
        
        print(f"  Processed {len(data)} valid examples")
        return data
        
    except Exception as e:
        print(f"  Error loading empathetic dialogues: {e}")
        return []

def load_daily_dialog(file_path, split_name):
    """Load and process DailyDialog CSV files (train/test/validation)."""
    print(f"\nLoading DailyDialog {split_name} from {file_path}...")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"  Loaded {len(df)} rows")
        print(f"  Columns: {df.columns.tolist()}")
        
        data = []
        
        for idx, row in df.iterrows():
            # DailyDialog has 'dialog' and 'emotion' columns
            dialog = row.get('dialog', '')
            emotion = row.get('emotion', 'neutral')
            
            # Parse dialog (it's a list-like string)
            if pd.notna(dialog) and isinstance(dialog, str):
                # Extract individual utterances from the dialog
                # Format is usually like: ['utterance1', 'utterance2', ...]
                try:
                    # Simple extraction - take the whole dialog as context
                    processed_text = preprocess_text(dialog)
                    if processed_text and len(processed_text) > 5:
                        data.append({
                            'text': processed_text,
                            'emotion': map_emotion_to_standard(emotion),
                            'source': f'daily_dialog_{split_name}'
                        })
                except:
                    pass
        
        print(f"  Processed {len(data)} valid examples")
        return data
        
    except Exception as e:
        print(f"  Error loading DailyDialog {split_name}: {e}")
        return []

def balance_dataset(data, max_per_emotion=5000):
    """Balance dataset to prevent class imbalance."""
    print(f"\nBalancing dataset (max {max_per_emotion} per emotion)...")
    
    # Group by emotion
    emotion_groups = {}
    for item in data:
        emotion = item['emotion']
        if emotion not in emotion_groups:
            emotion_groups[emotion] = []
        emotion_groups[emotion].append(item)
    
    # Print distribution before balancing
    print("  Distribution before balancing:")
    for emotion, items in sorted(emotion_groups.items()):
        print(f"    {emotion}: {len(items)}")
    
    # Balance by limiting each emotion
    balanced_data = []
    for emotion, items in emotion_groups.items():
        if len(items) > max_per_emotion:
            # Randomly sample
            import random
            random.seed(42)
            sampled = random.sample(items, max_per_emotion)
            balanced_data.extend(sampled)
        else:
            balanced_data.extend(items)
    
    # Print distribution after balancing
    emotion_counts = Counter([item['emotion'] for item in balanced_data])
    print("\n  Distribution after balancing:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"    {emotion}: {count}")
    
    return balanced_data

def main():
    """Main processing function."""
    print("="*60)
    print("DATASET PROCESSING PIPELINE")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, 'Dataset')
    output_dir = os.path.join(script_dir, 'data', 'training')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all datasets
    all_data = []
    
    # 1. Load Empathetic Dialogues (69k)
    empathetic_path = os.path.join(dataset_dir, 'emotion-emotion_69k.csv')
    if os.path.exists(empathetic_path):
        empathetic_data = load_empathetic_dialogues(empathetic_path)
        all_data.extend(empathetic_data)
    else:
        print(f"Warning: {empathetic_path} not found")
    
    # 2. Load DailyDialog datasets
    for split in ['train', 'test', 'validation']:
        daily_path = os.path.join(dataset_dir, f'{split}.csv')
        if os.path.exists(daily_path):
            daily_data = load_daily_dialog(daily_path, split)
            all_data.extend(daily_data)
        else:
            print(f"Warning: {daily_path} not found")
    
    print(f"\n{'='*60}")
    print(f"TOTAL EXAMPLES LOADED: {len(all_data)}")
    print(f"{'='*60}")
    
    # Balance dataset
    balanced_data = balance_dataset(all_data, max_per_emotion=5000)
    
    # Convert to training format
    training_data = []
    for item in balanced_data:
        training_data.append({
            'text': item['text'],
            'label': item['emotion']  # Use 'label' for consistency with existing code
        })
    
    # Save to JSON
    output_path = os.path.join(output_dir, 'merged_training_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(training_data)} training examples to:")
    print(f"  {output_path}")
    
    # Generate statistics report
    stats = {
        'total_examples': len(training_data),
        'emotion_distribution': dict(Counter([item['label'] for item in training_data])),
        'source_files': [
            'emotion-emotion_69k.csv',
            'train.csv',
            'test.csv',
            'validation.csv'
        ],
        'processing_date': pd.Timestamp.now().isoformat()
    }
    
    stats_path = os.path.join(output_dir, 'dataset_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Saved statistics report to:")
    print(f"  {stats_path}")
    
    print(f"\n{'='*60}")
    print("DATASET PROCESSING COMPLETE!")
    print(f"{'='*60}")
    
    return training_data

if __name__ == '__main__':
    main()
