from tokenizers import Tokenizer
import json

def analyze_tokenizer():
    try:
        tokenizer = Tokenizer.from_file("saslm_tokenizer.json")
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        return

    vocab = tokenizer.get_vocab()
    print(f"Vocab Size: {tokenizer.get_vocab_size()}")
    
    # Key terms in Sri Aurobindo's philosophy
    test_words = [
        "supramental", "supermind", "overmind", 
        "nescience", "inconscient", 
        "sadhana", "purusha", "prakriti", 
        "psychic", "involution", "evolution",
        "parabrahman", "saccidananda"
    ]
    
    print("\n--- Tokenization Test ---")
    print(f"{'Word':<20} | {'Tokens':<30} | {'Count'}")
    print("-" * 60)
    
    for word in test_words:
        # Encode
        encoded = tokenizer.encode(word)
        tokens = encoded.tokens
        
        # Check if it's a single token (or single token + prefix space)
        # Note: BPE often adds Ġ (space) if it's start of sentence or preceded by space
        # We test purely the word.
        
        print(f"{word:<20} | {str(tokens):<30} | {len(tokens)}")

    # Check fertility on a sample sentence
    sample = "The supramental consciousness is the inevitable consequence of the spiritual evolution."
    encoded_sample = tokenizer.encode(sample)
    print("\n--- Phrase Analysis ---")
    print(f"Sentence: '{sample}'")
    print(f"Tokens: {encoded_sample.tokens}")
    print(f"Token Count: {len(encoded_sample.tokens)}")
    print(f"Word Count: {len(sample.split())}")
    print(f"Fertility (Tokens/Word): {len(encoded_sample.tokens)/len(sample.split()):.2f}")

if __name__ == "__main__":
    analyze_tokenizer()
