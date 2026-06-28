from sentence_transformers import SentenceTransformer
import os

def main():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_model")
    print(f"Downloading model to {model_dir}...")
    # This downloads the model from HuggingFace
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # This saves it to the local_model directory
    model.save(model_dir)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
