from datasets import load_dataset
import sys

def main():
    try:
        print("Checking SahandNZ/cryptonews-articles-with-price-momentum-labels...")
        ds1 = load_dataset("SahandNZ/cryptonews-articles-with-price-momentum-labels", split="train[:1]")
        print(f"Columns: {ds1.column_names}")
        print(f"Row 0: {ds1[0]}")
        print("---")
        
        print("\nChecking zeroshot/twitter-financial-news-topic...")
        ds2 = load_dataset("zeroshot/twitter-financial-news-topic", split="train[:1]")
        print(f"Columns: {ds2.column_names}")
        print(f"Row 0: {ds2[0]}")
        print("---")
        print("SUCCESS: Datasets loaded successfully.")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
