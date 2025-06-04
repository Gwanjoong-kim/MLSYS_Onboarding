from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.cuda.nvtx as nvtx
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import cupy as cp

# data processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(
    root = "./data",
    train = True,
    download = True,
    transform = transform,
)

test_dataset = datasets.MNIST(
    root = "./data",
    train = False,
    download = True,
    transform = transform
)

# data loader
train_loader = DataLoader(
    train_dataset,
    batch_size = 1024,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 1024,
    shuffle = True
)

# 보조 함수들
def relu(x):
    return cp.where(x<0, 0, x)
    
def sigmoid(x):
    x_clipped = cp.clip(x, -50, 50)
    x = 1.0 / (1.0 + cp.exp(-x_clipped))
    return x

def softmax(x):
    exps = cp.exp(x)
    return exps / cp.sum(exps, axis = 1, keepdims = True)

def derivativeOfRelu(x):
    return cp.where(x<0, 0, 1)

def derivativeOfSigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))

def ce_loss(predictions, labels):
    # prediction: 64 * 10
    # labels: 64개의 true 값이었음 => 64 * 10, one-hot vector
    labels = cp.eye(10)[labels]
    ce = -cp.sum(labels * cp.log(predictions))/ labels.shape[0]
    return ce

def metric(prediction, labels):
    # predictions: 64 
    # labels: 64 
    acc = accuracy_score(prediction, labels)
    prec = precision_score(prediction, labels, average = 'macro', zero_division=0)
    rec = recall_score(prediction, labels, average = 'macro', zero_division=0)
    
    return (acc, prec, rec)
    
    
# DNN class
class DNN():
    def __init__(self, layer_sizes, learning_rate, mp_instances = None):
        # input size: BS * 1 * 28 * 28
        # layer_size: [28*28, 14*14, 8*8, 10]
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        self.mp_instances = mp_instances
        
        # Weight와 bias를 list 형태로 저장한 후 각 layer에서 불러오기
        for i in range(self.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            W = cp.random.randn(in_dim, out_dim)
            b = cp.zeros((1, out_dim))
            self.weights.append(W.astype(cp.float32))
            self.biases.append(b.astype(cp.float32))
    
    def forward(self, x):
        self.Zs = []
        self.As = [cp.asarray(x.numpy().astype(np.float32))]
                
        for l in range(self.num_layers - 1):
            Z = self.As[l] @ self.weights[l] + self.biases[l]
            if l == 0:
                A = relu(Z)
            else: 
                A = sigmoid(Z)
                
            self.Zs.append(Z)
            self.As.append(A)
        
        Z_last = self.As[-1] @ self.weights[-1] + self.biases[-1]
        A_last = softmax(Z_last)
        
        self.Zs.append(Z_last)
        self.As.append(A_last)
        
        return A_last

    # Backpropagation of loss
    def backward(self, y):
        bs = self.As[-1].shape[0]
        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers
        
        A_last = self.As[-1]
        dZ = (A_last - cp.eye(10)[y]) / bs
        
        grads_W[-1] = self.As[-2].T @ dZ
        grads_b[-1] = cp.sum(dZ, axis = 0, keepdims = True)
                
        for l in reversed(range(self.num_layers - 1)):
            dA = dZ @ self.weights[l + 1].T
            if (l == 0):
                dZ = dA * derivativeOfRelu(self.Zs[l])
            else:
                dZ = dA * derivativeOfSigmoid(self.Zs[l])
                
            grads_W[l] = self.As[l].T @ dZ
            grads_b[l] = cp.sum(dZ, axis = 0, keepdims = True)
                        
        return grads_W, grads_b
    
    def update_params(self, grad_W, grad_b):
        for l in range(self.num_layers):
            # pdb.set_trace()
            self.weights[l] -= self.learning_rate * grad_W[l]
            self.biases[l] -= self.learning_rate * grad_b[l]
            
    def train(self, epochs):
        
        self.per_epoch_train_acc = []
        self.per_epoch_train_loss = []
        
        for epoch in tqdm(range(epochs)):
            train_acc = []
            train_loss = []
            for iteration, data in enumerate(train_loader):
                
                nvtx.range_push(f"Forward: {epoch} epoch - {iteration} iteration")
                preds = self.forward(data[0])
                nvtx.range_pop()

                pred_label = np.argmax(preds, axis = 1)
                train_acc.append(metric(pred_label.get(), data[1])[0])
                train_loss.append(ce_loss(preds, data[1]).get())
                
                nvtx.range_push(f"Backward: {epoch} epoch - {iteration} iteration")
                grads_W, grads_b = self.backward(data[1])
                self.update_params(grads_W, grads_b)
                nvtx.range_pop()
                
            self.per_epoch_train_acc.append(sum(train_acc) / len(train_loader))
            self.per_epoch_train_loss.append(sum(train_loss) / len(train_loader))
            
        self.draw(epochs)
        self.eval()
        
    def eval(self):
        eval_metric = []   
        final_res = [0, 0, 0]
        for iteration, data in enumerate(test_loader):
            preds = self.forward(data[0], "eval", iteration)
            res = np.argmax(preds, axis = 1)
            eval_metric.append(metric(res.get(), data[1]))
            
        for data in eval_metric:
            final_res[0] += data[0]
            final_res[1] += data[1]
            final_res[2] += data[2]
        
        final_res[0] /= len(eval_metric)
        final_res[1] /= len(eval_metric)
        final_res[2] /= len(eval_metric)
        
        print(f"Accuracy: {final_res[0]} \nPrecision: {final_res[1]} \nRecall: {final_res[2]}")
            
    def draw(self, epochs):
        # --- 정확도 그래프 ---
        plt.figure()  # 새로운 Figure 생성
        # x축: 1부터 epochs까지, y축: self.train_acc 리스트
        
        plt.plot(range(1, epochs+1), self.per_epoch_train_acc, label="Train Accuracy", marker='o')
        plt.title("Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xticks(range(1, epochs+1))       # 에포크 번호(1,2,3,...,20)를 x축 눈금으로 표시
        plt.grid(linestyle='--', alpha=0.5)  # 가독성을 위해 점선 격자 추가
        plt.legend()                         # 범례 표시
        plt.savefig("acc_curve.png")         # 파일로 저장
        plt.clf()                            # 같은 Figure를 다시 쓰지 않도록 비움

        # --- 손실 그래프 ---
        plt.figure()
        plt.plot(range(1, epochs+1), self.per_epoch_train_loss, label="Train Loss", marker='x', color='red')
        plt.title("Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(range(1, epochs+1))
        plt.grid(linestyle='--', alpha=0.5)
        plt.legend()
        plt.savefig("loss_curve.png")
        plt.clf()
        

class pseudo_MP():
    def __init__(self, num_of_gpus, model):
        self.num_of_gpus = num_of_gpus
        self.model = model
    
    def copy_model(self, model):
        
        # Multi GPU에 Model을 모두 복사함
        for gpu in range(self.num_of_gpus):
            model.to(gpu)
            
        return 0
    
    def forward_per_gpu(self):
        # 각 GPU가 sub batch 분량의 data에 forward 연산을 독립적으로 
        # model의 forward 파트를 불러와서 이용함
        return 0
    
    def communicates_over_gpu(self):
        # 각 GPU가 계산한 Gradients들을 CPU나 Main host GPU로 전송함
        # 각 GPU에서 모인 Gradients들을 평균내어서 다시 각 GPU에 전송함
        return 0
    
    def backward_per_gpu(self):
        # communicates_over_gpu 함수에서 전달받은 forward gradients들을 
        # 각 GPU에서 각 sub batch 분량의 data에 backward 연산을 독립적으로 연산함
        # model의 backward 파트를 불러와서 이용함
        return 0
    
    
model = DNN([28*28, 14*14, 8*8, 10], 0.1)

mp = pseudo_MP(4, model)
model.mp_instances = mp
model.mp_instances.copy_model(model)
model.mp_instances.forward_per_gpu()
model.mp_instances.communicates_over_gpu()
model.mp_instances.backward_per_gpu()

# DNN에서 정의한 training loop에 위의 MP를 반영함

model.train(10)