# useModel.py - Clean inference script
import tiktoken
import torch
from model_classes import GPT, GPTConfig

def load_model(model_path="best_model_params.pt"):
    """Load the trained GPT model"""
    # Initialize model configuration (must match training config)
    config = GPTConfig(
        vocab_size=50257,
        block_size=128,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,  # This will be ignored in eval mode
        bias=True
    )
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    
    # Create model and load weights
    model = GPT(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"📦 Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, device

def generate_text(model, device, prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt"""
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            tokens, 
            max_tokens, 
            temperature=temperature, 
            top_k=top_k
        )
    
    # Decode and return
    return enc.decode(generated[0].tolist())

# Main usage
if __name__ == "__main__":
    # Load model
    model, device = load_model("best_model_params.pt")
    
    # Test generation
    test_prompt = "Tell me a stupid story without girls name Lily"
    print(f"\n📝 Generating text...")
    print(f"Prompt: '{test_prompt}'")
    
    generated_text = generate_text(
        model, 
        device, 
        test_prompt, 
        max_tokens=100, 
        temperature=0.8, 
        top_k=50
    )
    
    print(f"\n🎯 Generated text:")
    print(generated_text)
    
    # Interactive mode
    print(f"\n🎮 Interactive mode (type 'quit' to exit):")
    while True:
        user_prompt = input("\nEnter prompt: ").strip()
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_prompt:
            result = generate_text(model, device, user_prompt, max_tokens=80)
            print(f"\nGenerated: {result}")
    
    print("👋 Goodbye!")