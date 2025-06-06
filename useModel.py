# useModel.py - Fixed inference script
import tiktoken
import torch
import json
from model_class import GPT, GPTConfig

def load_model(model_path="gpt_model_complete.pt", config_path="model_config.json"):
    """Load the trained GPT model"""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    
    # Load config from JSON file
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = GPTConfig(**config_dict)
        print(f"📋 Loaded config from {config_path}")
        print(f"   - Vocab size: {config.vocab_size:,}")
        print(f"   - Block size: {config.block_size}")
        print(f"   - Layers: {config.n_layer}")
        print(f"   - Heads: {config.n_head}")
        print(f"   - Embedding dim: {config.n_embd}")
    except FileNotFoundError:
        print(f"⚠️  Config file {config_path} not found, using default config")
        config = GPTConfig(
            vocab_size=50257,
            block_size=128,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=0.1,
            bias=True
        )
    
    # Load the checkpoint
    try:
        # Method 1: Try with safe globals (recommended)
        with torch.serialization.safe_globals([GPTConfig]):
            checkpoint = torch.load(model_path, map_location=device)
    except:
        # Method 2: Fallback to weights_only=False (if you trust the file)
        print("⚠️  Using weights_only=False (fallback method)")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model
    model = GPT(config).to(device)
    
    # Load the model state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded model_state_dict from checkpoint")
    else:
        # If the checkpoint is just the state dict itself
        model.load_state_dict(checkpoint)
        print("✅ Loaded state dict directly")
    
    model.eval()
    
    print(f"📦 Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print some checkpoint info if available
    if isinstance(checkpoint, dict):
        if 'best_val_loss' in checkpoint:
            print(f"📊 Best validation loss: {checkpoint['best_val_loss']:.4f}")
        if 'train_loss_history' in checkpoint and checkpoint['train_loss_history']:
            print(f"📈 Final training loss: {checkpoint['train_loss_history'][-1]:.4f}")
    
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
    model, device = load_model("gpt_model_complete.pt", "model_config.json")
    
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