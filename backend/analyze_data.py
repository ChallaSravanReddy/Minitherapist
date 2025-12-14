import pandas as pd
import os

def analyze_dataset():
    file_path = r'C:\Code\Projects\Minitherapist\Dataset\emotion-emotion_69k.csv'
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        print("Columns:", df.columns.tolist())
        print("\nShape:", df.shape)
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nUnique Emotions:")
        if 'emotion' in df.columns:
            print(df['emotion'].value_counts())
            
        print("\nSample Dialogue:")
        if 'empathetic_dialogues' in df.columns:
            print(df['empathetic_dialogues'].iloc[0])
            
    except Exception as e:
        print(f"Error reading CSV: {e}")

if __name__ == "__main__":
    analyze_dataset()
