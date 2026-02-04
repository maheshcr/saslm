import torch
import torch.nn.functional as F
from tokenizers import Tokenizer as HFTokenizer
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train_saslm import GPT, GPTConfig
except ImportError:
    print("Error: Could not import GPT class from train_saslm.py. Make sure you are running this from the project root or src directory.")
    sys.exit(1)

def load_model(model_path, tokenizer_path):
    print(f"Loading tokenizer from {tokenizer_path}...")
    if not os.path.exists(tokenizer_path):
         print(f"Error: Tokenizer not found at {tokenizer_path}")
         return None, None, None
    
    tokenizer = HFTokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    # Initialize Model
    # These parameters must match the training configuration exactly
    # WARN: The checkpoint was trained with vocab_size=30000.
    # If local tokenizer is smaller, we must pad or force the model config to 30000 to load weights.
    model_vocab_size = 30000 
    print(f"Initializing model with vocab_size={model_vocab_size} (Checkpoint requirement)...")
    if vocab_size != model_vocab_size:
        print(f"WARNING: Local tokenizer vocab size ({vocab_size}) != Model checkpoint vocab size ({model_vocab_size}).")
        print("Text generation might be effectively gibberish if the token mappings don't align.")

    config = GPTConfig(vocab_size=model_vocab_size, block_size=512) 
    model = GPT(config)
    
    # Load Weights
    print(f"Loading weights from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None, None, None

    # Auto-detect device (prioritize MPS for Mac)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print(f"Using device: {device.upper()}")
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load state dict: {e}")
        return None, None, None

    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def generate_text(model, tokenizer, device, prompt, max_tokens=100, temp=0.8, top_k=40):
    # Encode
    encoded = tokenizer.encode(prompt)
    if not encoded.ids:
        return ""
        
    input_ids = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        # Call the generate method from the GPT class
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temp, top_k=top_k)
        
    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    return generated_text

def run_yogic_turing_test(model, tokenizer, device):
    prompts = [
        "The distinction between the psychic being and the spiritual self is",
        "The three lower planes of existence are Matter, Life, and",
        "The power that mediates between the Supermind and the Mind is the",
        "The goal of Integral Yoga is not Nirvana, but",
        "In the process of involution, the Divine Consciousness hides itself in"
    ]
    
    print("\n=== Running Yogic Turing Test Prompts ===")
    for p in prompts:
        print(f"\nPrompt: {p}")
        print("Model: ", end="", flush=True)
        res = generate_text(model, tokenizer, device, p, max_tokens=60)
        print(res.replace(p, "...") if p in res else res) # minimal formatting
        print("-" * 30)

if __name__ == "__main__":
    # Default paths based on user info
    MODEL_PATH = "models/saslm_model-v0.pth"
    TOKENIZER_PATH = "saslm_tokenizer.json"
    
    # Allow overriding via args
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    
    model, tokenizer, device = load_model(MODEL_PATH, TOKENIZER_PATH)
    
    if model:
        # Run standard test
        run_yogic_turing_test(model, tokenizer, device)
        
        # Interactive Mode
        print("\n=== Interactive Mode (Type 'quit' to exit) ===")
        while True:
            prompt = input("\nEnter prompt: ")
            if prompt.lower() in ['quit', 'exit']:
                break
            
            print("Generating...", end="", flush=True)
            res = generate_text(model, tokenizer, device, prompt, max_tokens=100)
            print(f"\n{res}\n")
