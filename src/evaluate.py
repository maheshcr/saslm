import argparse
import json
import time
import os
import sys
import datetime
import csv
import torch
from tqdm import tqdm

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import load_model, generate_text
try:
    from llm_judge import LLMJudge
except ImportError:
    LLMJudge = None

def setup_results_dir():
    if not os.path.exists("eval_results"):
        os.makedirs("eval_results")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/saslm_model-v1.pth")
    parser.add_argument("--tokenizer_path", type=str, default="saslm_tokenizer.json")
    parser.add_argument("--prompts_file", type=str, default="eval_data/yogic_turing_prompts.json")
    parser.add_argument("--judge_provider", type=str, default=None, help="openai, anthropic, or gemini. If None, inference only.")
    parser.add_argument("--max_gen_tokens", type=int, default=100)
    args = parser.parse_args()

    setup_results_dir()
    
    # Load Model
    print(f"Loading model from {args.model_path}...")
    model, tokenizer, device = load_model(args.model_path, args.tokenizer_path)
    if not model:
        print("Model load failed.")
        return

    # Load Prompts
    with open(args.prompts_file, 'r') as f:
        prompts = json.load(f)
        
    # Init Judge
    judge = None
    if args.judge_provider:
        if LLMJudge:
            print(f"Initializing Judge with {args.judge_provider}...")
            judge = LLMJudge(provider=args.judge_provider)
        else:
            print("Warning: llm_judge module not imported correctly.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"eval_results/eval_{timestamp}.csv"
    
    results = []
    
    print(f"Running evaluation on {len(prompts)} prompts...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'category', 'prompt', 'generated_text', 'latency_ms', 
                      'ontology_score', 'style_score', 'coherence_score', 'overall_score', 'judge_reasoning']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in tqdm(prompts):
            p_text = item['prompt']
            
            start_t = time.time()
            gen_text = generate_text(model, tokenizer, device, p_text, max_tokens=args.max_gen_tokens)
            end_t = time.time()
            latency = int((end_t - start_t) * 1000)
            
            # Remove the prompt from the answer if it repeats it? 
            # generate_text usually includes prompt.
            # But the judge might want to see the full flow.
            # Let's keep it as is.
            
            row = {
                'id': item['id'],
                'category': item['category'],
                'prompt': p_text,
                'generated_text': gen_text,
                'latency_ms': latency,
                'ontology_score': 0,
                'style_score': 0,
                'coherence_score': 0,
                'overall_score': 0,
                'judge_reasoning': ""
            }
            
            # Judge
            if judge:
                # We strip the prompt from gen_text passed to judge?
                # The judge prompt format in llm_judge.py expects 'generated completion'.
                # Ideally we pass only the new tokens. 
                # Our generate_text returns full sequence.
                completion_only = gen_text[len(p_text):]
                
                # Grade
                grade = judge.grade_response(p_text, completion_only)
                if grade:
                    row['ontology_score'] = grade.get('ontological_accuracy', 0)
                    row['style_score'] = grade.get('stylistic_fidelity', 0)
                    row['coherence_score'] = grade.get('coherence', 0)
                    row['overall_score'] = grade.get('overall_score', 0)
                    row['judge_reasoning'] = grade.get('reasoning', "")
            
            writer.writerow(row)
            results.append(row)
            csvfile.flush() # ensure data is written if crash

    print(f"Evaluation complete. Results saved to {output_file}")
    
    # Calculate average score if judge was active
    if judge and results:
        avg_score = sum(r['overall_score'] for r in results) / len(results)
        print(f"Average Overall Score: {avg_score:.2f} / 10")

if __name__ == "__main__":
    main()
