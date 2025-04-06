from speechbrain.inference.separation import SepformerSeparation
import torch

# Initialize the SepFormer model
model = SepformerSeparation.from_hparams(source='speechbrain/sepformer-wsj02mix')

# Print the named modules
print("Potential LoRA target modules:")
print("=" * 50)

# Find all Linear layers (potential LoRA targets)
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
        linear_layers.append((name, module.__class__.__name__, 
                             module.in_features if isinstance(module, torch.nn.Linear) else module.in_channels,
                             module.out_features if isinstance(module, torch.nn.Linear) else module.out_channels))

# Find attention modules
print("Attention Modules:")
for name, module in model.named_modules():
    if 'att' in name.lower() and 'attention' in module.__class__.__name__.lower():
        print(f"{name}: {module.__class__.__name__}")

print("\nLinear Layers:")
print("{:<60} {:<20} {:<10} {:<10}".format("Name", "Type", "In", "Out"))
print("-" * 100)
for name, module_type, in_features, out_features in linear_layers:
    print("{:<60} {:<20} {:<10} {:<10}".format(name, module_type, in_features, out_features))

# Find potential module patterns we can target with LoRA
print("\nPotential Target Module Patterns:")
attention_patterns = set()
for name, _ in model.named_modules():
    if 'att' in name.lower():
        # Get the pattern by removing the numbers and specific module names
        pattern = '.'.join([part for part in name.split('.') if not part.isdigit() and part not in ['in_proj_weight', 'out_proj']])
        attention_patterns.add(pattern)

for pattern in sorted(attention_patterns):
    print(f"- {pattern}")

# Examine module parameter names to find query, key, value projections
print("\nParameter Names (looking for q_proj, k_proj, v_proj):")
for name, _ in model.named_parameters():
    if any(x in name for x in ['query', 'key', 'value', 'q_', 'k_', 'v_', 'qkv']):
        print(f"- {name}")

# Print all parameters to look for patterns
print("\nAll Parameter Names:")
for name, _ in model.named_parameters():
    print(f"- {name}") 