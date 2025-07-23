import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BasicBlock(nn.Module):
    expansion = 1


    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )


    def forward(self, x): 
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = self.out_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.in_channels, num_classes)


    def _make_layer(self, block, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for stride_val in strides:
            layers.append(block(self.in_channels, self.out_channels, stride_val))
            self.in_channels = self.out_channels * block.expansion

        self.out_channels *= 2
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out



# --- Function to create ResNet-18 ---
def ResNet18(num_classes=10):
    """
    Instantiates a ResNet-18 model.
    num_blocks = [2, 2, 2, 2] means 2 BasicBlocks in each of the four main layers.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

if __name__ == '__main__':
    # =======================================================
    # Model Initialization and Forward Pass Sanity Check
    # =======================================================
    dummy_input = torch.randn(1, 3, 32, 32)
    model = ResNet18(num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    dummy_input = dummy_input.to(device)

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        output = model(dummy_input)

    print(f"--- Model Sanity Check ---")
    print(f"Model: ResNet-18")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape (logits): {output.shape}") 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params / 1e6:.2f} Million") 
    print(f"")

    # =======================================================
    # Basic Training Loop Sanity Check
    # =======================================================
    print(f"--- Training Loop Sanity Check ---")
    
    model.train() # Set model to training mode (important for BatchNorm)

    criterion = nn.CrossEntropyLoss()
    # FIX: Changed optimizer to Adam, which is often more robust to small/sparse gradients
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Common learning rate for Adam

    batch_size = 32 
    dummy_train_input = torch.randn(batch_size, 3, 32, 32).to(device)
    dummy_train_labels = torch.randint(0, 10, (batch_size,)).to(device) 

    initial_param_value = model.linear.weight.data.clone().sum()
    print(f"Initial sum of linear layer weights: {initial_param_value.item():.4f}")

    optimizer.zero_grad() 
    outputs = model(dummy_train_input)
    loss = criterion(outputs, dummy_train_labels)
    print(f"Loss for dummy batch: {loss.item():.4f}")

    loss.backward()

    nan_inf_found = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"WARNING: Gradient of {name} contains NaN!")
                nan_inf_found = True
            if torch.isinf(param.grad).any():
                print(f"WARNING: Gradient of {name} contains Inf!")
                nan_inf_found = True
    if nan_inf_found:
        print("Gradient NaN/Inf detected! Parameters might not update.")
    else:
        print("No NaN/Inf gradients detected. (Good!)")

    # ADDITION: Inspect linear layer's gradients more thoroughly
    if model.linear.weight.grad is not None:
        print(f"linear.weight gradients (sum): {model.linear.weight.grad.sum().item():.4f}")
        print(f"linear.weight gradients (max abs): {model.linear.weight.grad.abs().max().item():.4f}")
    else:
        print("linear.weight has no gradients. (This might indicate an issue.)")

    if model.conv1.weight.grad is not None:
        print(f"conv1.weight gradients (sum): {model.conv1.weight.grad.sum().item():.4f}")
        print(f"conv1.weight gradients (max abs): {model.conv1.weight.grad.abs().max().item():.4f}")
    else:
        print("conv1.weight has no gradients. (This might indicate an issue.)")

    optimizer.step()

    final_param_value = model.linear.weight.data.clone().sum()
    print(f"Final sum of linear layer weights: {final_param_value.item():.4f}")

    if not torch.allclose(initial_param_value, final_param_value):
        print(f"Parameters updated successfully! (Initial != Final)")
    else:
        print(f"Parameters did NOT update. (This indicates an issue!)")

    print(f"Training loop sanity check complete.")
