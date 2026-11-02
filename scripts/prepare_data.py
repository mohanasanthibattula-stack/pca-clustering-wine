"""Extract and save the UCI Wine dataset from scikit-learn unchanged."""
import pandas as pd
from sklearn.datasets import load_wine

def main():
    wine = load_wine(as_frame=True)
    df = wine.frame.copy()
    # rename target column to cultivar as specified
    if 'target' in df.columns:
        df = df.rename(columns={'target': 'cultivar'})
    
    # Verify shape and class counts
    print(f"Shape: {df.shape}")
    print(f"Cultivar counts:\n{df['cultivar'].value_counts().sort_index()}")
    
    # Save to data/wine.csv
    df.to_csv("data/wine.csv", index=False)
    print("Saved to data/wine.csv successfully.")

if __name__ == "__main__":
    main()
