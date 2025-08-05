import pickle

# Load your chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Number of chunks:", len(chunks))
print("First chunk sample:", chunks[0][:500])  # Show first 500 characters
