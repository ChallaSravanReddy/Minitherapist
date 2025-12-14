import pandas as pd
import json
import os

def process_dataset():
    input_path = r'C:\Code\Projects\Minitherapist\Dataset\emotion-emotion_69k.csv'
    output_path = r'C:\Code\Projects\Minitherapist\backend\data\training\large_emotion_dataset.json'
    
    # Emotion mapping dictionary
    emotion_map = {
        # SAD
        'sad': 'sad', 'devastated': 'sad', 'disappointed': 'sad', 
        'embarrassed': 'sad', 'ashamed': 'sad', 'guilty': 'sad', 
        'nostalgic': 'sad', 'sentimental': 'sad',
        
        # STRESSED
        'anxious': 'stressed', 'apprehensive': 'stressed', 
        'overwhelmed': 'overwhelmed', # Keep overwhelmed separate if possible, or map to stressed
        
        # ANGRY
        'angry': 'angry', 'furious': 'angry', 'annoyed': 'angry', 
        'jealous': 'angry', 'disgusted': 'angry',
        
        # LONELY
        'lonely': 'lonely',
        
        # CONFUSED
        'surprised': 'confused', # Closest fit
        
        # HAPPY
        'happy': 'happy', 'joyful': 'happy', 'proud': 'happy', 
        'grateful': 'happy', 'excited': 'happy', 'confident': 'happy', 
        'content': 'happy', 'impressed': 'happy', 'trusting': 'happy', 
        'caring': 'happy', 'faithful': 'happy', 'prepared': 'happy', 
        'hopeful': 'happy', 'anticipating': 'happy',
        
        # WORRIED
        'afraid': 'worried', 'terrified': 'worried'
    }
    
    try:
        print("Reading CSV...")
        df = pd.read_csv(input_path)
        
        training_data = []
        skipped_count = 0
        
        print("Processing rows...")
        for index, row in df.iterrows():
            original_emotion = str(row['emotion']).strip().lower()
            text = str(row['Situation']).strip()
            
            # Skip if text is empty or too short
            if len(text) < 3:
                continue
                
            # Map emotion
            if original_emotion in emotion_map:
                mapped_emotion = emotion_map[original_emotion]
                training_data.append({
                    "text": text,
                    "label": mapped_emotion
                })
            else:
                # print(f"Skipping unknown emotion: {original_emotion}")
                skipped_count += 1
        
        print(f"\nProcessed {len(training_data)} examples.")
        print(f"Skipped {skipped_count} examples (unknown emotions).")
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2)
            
        print(f"Saved to {output_path}")
        
        # Print distribution
        dist = {}
        for item in training_data:
            label = item['label']
            dist[label] = dist.get(label, 0) + 1
        print("\nEmotion Distribution:")
        for k, v in dist.items():
            print(f"{k}: {v}")
            
    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    process_dataset()
