from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import glob
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_tokenizer(vocab_size=32000):
    """
    Trains a BPE tokenizer on the .txt files in processed_text/.
    """
    files = glob.glob("processed_text/*.txt")
    if not files:
        logging.error("No text files found in processed_text/. Run extraction pipeline first.")
        return

    logging.info(f"Found {len(files)} files to train on.")

    # Initialize Tokenizer in BPE
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # Pre-tokenizer: Split by whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Decoder: Reconstruct from ByteLevel
    tokenizer.decoder = decoders.ByteLevel()
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        min_frequency=2,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # Train
    logging.info("Starting tokenizer training...")
    tokenizer.train(files, trainer)
    
    # Post-processor (for BERT-like wrapping if needed, but for GPT usually simple)
    # GPT2 uses ByteLevel, no explicit post-processor beyond that usually needed for generation loop.
    
    # Save
    save_path = "saslm_tokenizer.json"
    tokenizer.save(save_path)
    logging.info(f"Tokenizer saved to {save_path}")
    
    # Test
    sample = "The nature of the Supermind is rta-chit."
    encoded = tokenizer.encode(sample)
    logging.info(f"Test Sentence: '{sample}'")
    logging.info(f"Tokens: {encoded.tokens}")
    logging.info(f"IDs: {encoded.ids}")

if __name__ == "__main__":
    train_tokenizer()
