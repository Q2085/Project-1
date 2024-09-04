import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK TC'  # 設定繪圖字體

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### set a random seed for reproducibility
myseed = 42061
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(myseed)

''' **********************************
   1. 讀取資料，建立輸入 X 與對應的輸出 Y  
  '''

### 讀取 TCCIP 資料
t0 = (datetime(1980, 1, 1) - datetime(1960, 1, 1)).days
prec = nc.Dataset("TCCIP_GriddedRain_TWN.nc").variables['PREC_5km'][t0:]
dates = np.loadtxt('FTdate_56_f1980.txt', dtype=str)
dates = np.array([datetime.strptime(date_str, "%Y%m%d") # + timedelta(days=1) #TODO: 可設定改成預測 n 天以後的降雨
                  for date_str in dates]) 

days = (dates - datetime(1980, 1, 1)).astype('timedelta64[D]').astype(int)
prec = prec[days]


### 讀取 ERA5 資料
X = np.load('vars_hr_f1980.npz')['var']

### Normalizing ERA5 Data
X = (X - np.mean(X, axis=(0,2,3))[np.newaxis, :, np.newaxis, np.newaxis]) / np.std(X, axis=(0,2,3))[np.newaxis, :, np.newaxis, np.newaxis]

### 標記資料 Label：[0, 1, 2, 3]
th = [40, 80, 145]
max_prec = np.max(prec, axis=(1,2))
y = (max_prec>th[0]).astype(int) + (max_prec>th[1]).astype(int) + (max_prec>th[2]).astype(int)
nclass = np.max(y) + 1

### 可以查看最大降雨量的數量分布
# plt.figure(dpi=200)
# plt.plot(np.arange(940)/940, sorted(max_prec))
# [plt.axvline(np.argmin(abs(np.sort(max_prec)-th[i]))/940,
#               ymax = th[i]/np.max(max_prec),
#               color='k', linewidth=0.3) for i in range(3)]
# [plt.axhline(th[i], 
#               xmax = np.argmin(abs(np.sort(max_prec)-th[i]))/940,
#               color='k', linewidth=0.3) for i in range(3)]
# plt.xlim((0,1))
# plt.ylim(0)
# plt.ylabel("Rainfall [mm]")
# plt.title('Maximum Rainfall distribution [940#]', loc='left', fontsize=10)

count = [np.sum(y==i) for i in range(nclass)]
print(f'<{th[0]}mm / {th[0]}-{th[1]}mm / {th[1]}-{th[2]}mm / >{th[2]}mm\n'
      f'各類別數量：{count}\n')

#%%


''' **********************************
   2. 定義 CNN 模型
  '''

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            # Conv2d(in_channel, out_channel, kernel_size, Stride, Padding)
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(np.shape(X)[1], 10, 5, stride=1, padding=2),
            nn.ReLU(),
            )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(10, 12, 5, stride=1, padding=2),
            nn.ReLU(),
            # nn.Conv2d(10, 12, 7, stride=1, padding=3),
            # nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(12, 12, 9, stride=1, padding=4),
            nn.ReLU(),
            )
        self.fc = nn.Sequential(
            nn.Linear(12*20*17, nclass)
            )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(-1, 12*20*17)
        x = self.fc(x)
        return x

class GetDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def train(model, dataloader):
    model.train()
    
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x.float()).squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()   
    return loss.item()

def dev(model, dataloader):
    model.eval()
    dev_loss, accuracy = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            output = model(x.float())
            dev_loss += criterion(output, y).item()
            _, predicted = torch.max(output, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(torch.argmax(y, dim=1).cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    return dev_loss, accuracy, all_preds, all_labels

def plot_loss(loss, dev_loss, ax, i):
    global epochs, k_folds
    ax.plot(range(epochs), loss, range(epochs), dev_loss)
    ax.set_ylim((0.4, 1.5))
    ax.legend(['Training loss', 'Validation loss'])
    ax.set_title(f'Fold {i+1}', loc='left')
    if (i == k_folds-1):
        plt.suptitle(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}", fontsize=17)
        plt.tight_layout()
  
    
  
''' **********************************
   3. 進行訓練，並查看結果
  '''
  
### TODO: Model setting
batch_size = 15
lr = 2e-5
epochs = 36
k_folds = 4

X = torch.FloatTensor(X)
y_hot = F.one_hot(torch.tensor(y.astype(np.int64)), num_classes=nclass).to(torch.float32)

kf = KFold(n_splits=k_folds, shuffle=False)

fold_accuracies = []
conf_matrices = []
all_Y_true, all_Y_pred = [], []
all_loss, all_dev_loss = [], []

fig, ax = plt.subplots(2,2, figsize=(8,6), dpi=200)
for fold, (train_idx, dev_idx) in enumerate(kf.split(X)):
    print(f'Fold {fold+1}/{k_folds}')
    
    X_train, X_dev = X[train_idx], X[dev_idx]
    y_train, y_dev = y_hot[train_idx], y_hot[dev_idx]
    
    train_dataset = GetDataset(X_train, y_train)
    dev_dataset = GetDataset(X_dev, y_dev)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    dev_loader = DataLoader(dev_dataset, batch_size=len(y_dev))
    
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
    
    loss, dev_loss, dev_acc = [np.empty(epochs) for _ in range(3)]
    
    for epoch in range(epochs):
        loss[epoch] = train(model, train_loader)
        if (epoch+1) % (epochs // 5) == 0: 
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss[epoch]:.4f}")
        dev_loss[epoch], dev_acc[epoch], predicted, actual = dev(model, dev_loader)
    
    fold_accuracies.append(dev_acc[-1])
    conf_matrices.append(confusion_matrix(actual, predicted, labels=[0, 1, 2, 3]))
    all_Y_true.extend(actual.tolist())
    all_Y_pred.extend(predicted.tolist())
    
    print(f'Max Accuracy: {np.max(dev_acc):.4f}, {np.argmax(dev_acc)}')
    print(f'Fold Accuracy: {round(dev_acc[-1], 3)}\n')
    plot_loss(loss, dev_loss, ax[fold//2][fold%2], fold)
    all_loss.append(loss)
    all_dev_loss.append(dev_loss)
    
print(f'Average Accuracy over {k_folds} folds: {np.mean(fold_accuracies):.4f}')


### Plot the confusion matrix
c_matrix = np.sum(conf_matrices, axis=0)
# c_matrix = conf_matrices[0]

fig, ax = plt.subplots(dpi=250)
im = ax.imshow(c_matrix, cmap='Blues', vmax=160)
plt.colorbar(im)
ticks = ['<%imm'%th[0], '%i-%imm'%(th[0], th[1]), '%i-%imm'%(th[1], th[2]), '>%imm'%th[2]]
ax.set_xticks(np.arange(0, 4), ticks, fontsize=9)
ax.set_yticks(np.arange(0, 4), ticks, fontsize=9)

c_matrix = c_matrix.astype(int)
for i in range(4):
    for j in range(4):
        c = "w" if (c_matrix[i, j] / np.max(c_matrix)) > 0.6 else "k"
        text = ax.text(j, i, c_matrix[i, j],
                       ha="center", va="center", color=c)
        
plt.title(f"Confusion Matrix [{np.sum(c_matrix)}#]", loc='left', fontweight=600, fontsize=9)
plt.xlabel("預測", fontweight='500')
plt.ylabel("實際", fontweight='500')

from sklearn.metrics import matthews_corrcoef
print("MCC : ", matthews_corrcoef(all_Y_true, all_Y_pred))

#%%

### 輸出預測結果
opt = np.transpose(np.vstack((all_Y_true, all_Y_pred)))
if input("Continue to export? (y/n)\n") == 'y':
    torch.save(model.state_dict(), 'CNN_0824.pth')
    np.savetxt('Y_true_pred_f1980.txt', opt, fmt='%i', 
                header = f'True\tPred\t\t{np.mean(fold_accuracies):.4f}')

#%%

''' **********************************
   4. 透過 SHAP 套件計算 Feature Attribution並輸出
  '''

import shap
import random
random.seed(0)

background = X_train[:].to(device)
e = shap.GradientExplainer(model, background)
# e = shap.DeepExplainer(model, background)

n = 940  # TODO: 選擇要使用多少樣本進行全市
sample = random.sample(range(940), n)
test_images = X[sample]

true, pred = np.loadtxt('Y_true_pred_f1980.txt', skiprows=1, unpack=True, dtype=int)
dates = np.loadtxt('FTdate_56_f1980.txt', dtype=str)
dates = dates[sample]

shap_all = np.zeros((n, 4, 81, 69, 31))
for i, test_image in enumerate(test_images):
    shap_values = e.shap_values(test_image[np.newaxis, :, :, :])
    shap_values = np.array([np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]).squeeze()
    shap_all[i] = shap_values
    if(i % 10 == 0): print(f'{i} Completed')

if input("Continue to export? (y/n)\n") == 'y':
    np.savez('shap_f1980.npz', shap=shap_all)
    np.savetxt('shap_samples.txt', sample, fmt='%i')

