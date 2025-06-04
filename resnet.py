import torch
import torch.cuda.nvtx as nvtx
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

    
class ResNetPerKernelNVTX(models.ResNet):
    def __init__(self, *args, **kargs):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], *args, **kargs)
    
    def forward(self, x):
        
        nvtx.range_push("Conv1")
        x = self.conv1(x)
        nvtx.range_pop()
        
        nvtx.range_push("BN1")
        x = self.bn1(x)
        nvtx.range_pop()

        nvtx.range_push("ReLU1")
        x = self.relu(x)
        nvtx.range_pop()

        nvtx.range_push("MaxPool")
        x = self.maxpool(x)
        nvtx.range_pop()
        
        layer_counts = [3, 4, 6, 3]
        
        for i, num_blocks in enumerate(layer_counts):
            layer_module = getattr(self, f"layer{i+1}")
            
            for j in range(num_blocks):
                block = layer_module[j]
                
                # 1) Conv1
                nvtx.range_push(f"layer{i+1}_block{j}_conv1")
                x = block.conv1(x)
                nvtx.range_pop()

                # 2) BN1
                nvtx.range_push(f"layer{i+1}_block{j}_bn1")
                x = block.bn1(x)
                nvtx.range_pop()

                # 3) ReLU (첫 번째)
                nvtx.range_push(f"layer{i+1}_block{j}_relu1")
                x = block.relu(x)
                nvtx.range_pop()

                # 4) Conv2
                nvtx.range_push(f"layer{i+1}_block{j}_conv2")
                x = block.conv2(x)
                nvtx.range_pop()

                # 5) BN2
                nvtx.range_push(f"layer{i+1}_block{j}_bn2")
                x = block.bn2(x)
                nvtx.range_pop()

                # 6) ReLU (두 번째)
                nvtx.range_push(f"layer{i+1}_block{j}_relu2")
                x = block.relu(x)
                nvtx.range_pop()

                # 7) Conv3
                nvtx.range_push(f"layer{i+1}_block{j}_conv3")
                x = block.conv3(x)
                nvtx.range_pop()

                # 8) BN3
                nvtx.range_push(f"layer{i+1}_block{j}_bn3")
                x = block.bn3(x)
                nvtx.range_pop()
                
        return x
        

device = torch.device("cuda")
model = ResNetPerKernelNVTX()
model.to(device)

for i in range(2, 12, 1):

    batch_size = 2**i

    data = torch.randn([batch_size, 3, 224, 224], device = device)
    
    nvtx.range_push(f"Warmup (BS={batch_size})")
    with torch.no_grad():
        _ = model(data)
        torch.cuda.synchronize()
    nvtx.range_pop()
    
    nvtx.range_push(f"Inference (BS={batch_size})")
    with torch.no_grad():
        output = model(data)
        torch.cuda.synchronize()
    nvtx.range_pop()
    
    print("batch finish", batch_size)
    