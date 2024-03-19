import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from libs.Train_Gamma import Train_NN
from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error 


class ESN(nn.Module):
    "Implements the base ESN class"
    def __init__(self, sdim, spectral_radius=0.9, leaky_rate=0.3):
        
        super(ESN, self).__init__()
        self.activation = nn.Tanh()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dim = sdim
        self.spectral_radius = spectral_radius
        self.leaky_rate = leaky_rate
        print(self.device)
        self.W = torch.from_numpy(np.loadtxt('libs/W.txt')).float().to(self.device)
        #self.W = torch.rand(self.dim, self.dim, requires_grad=False, device = self.device)-0.5

        self.W = self.W/torch.max(torch.abs(torch.linalg.eigvalsh(self.W)))
        self.W_in = (torch.rand(self.dim, self.dim, requires_grad=False, device = self.device)-0.5)*2
        self.model_h = None
        self.model_r = None
        self.pca = None

        

        

    def forward(self, x, u):
        u = torch.from_numpy(np.pad(u.detach().cpu().numpy(), (0, self.dim-1), 'constant', constant_values=(0, 0))).float().to(self.device)
        return (1-self.leaky_rate)*x + self.leaky_rate*self.activation(self.spectral_radius*self.W @ x + self.W_in @ u)
    
    def collect_states(self, u, length, discard):
        x = torch.zeros(self.dim, device=self.device)
        states = []
        for i in range(length+discard):
            x = self.forward(x, u[i])
            if i > discard:
                states.append(x)
            if i%100 == 0:
                print('Collecting states: ', i, '/', length+discard)    
        return torch.stack(states).T
    
    def fit(self, u, length, discard, k):
        #readout
        states = self.collect_states(u, length, discard)

        self.model_r = Train_NN(states.T.to(self.device),u[discard+2:discard+length+1].to(self.device),hidden_layers=4,layer_dimension=500,epochs=500,batch_size=500)
        #h'
        # collect u in delays [u_n-k, u_n-k+1, ..., u_n-1, u_n]
        u_delays = torch.zeros(length-1, k)
        for i in range(k):
            u_delays[:,i] = u[discard-k+i+1:discard+length-k+i].reshape(-1)
        self.pca = PCA(n_components=100)
        self.scaler = MinMaxScaler(feature_range=(-0.2,0.2))
        y_train = torch.from_numpy(self.scaler.fit_transform(self.pca.fit_transform(states.T.detach().cpu().numpy()))).float()

        # train u_delays to x_n

        self.model_h = Train_NN(u_delays.to(self.device),y_train.to(self.device),hidden_layers=4,layer_dimension=500,epochs=500,batch_size=500)


        return states[:,-1]
    
    def predict(self, u, pred_len, warmup_len, k, err=0):
        # Warmup:
        prediction_w = torch.zeros(pred_len + warmup_len, 1, device=self.device)
        x = torch.zeros(self.dim, device = self.device)
        for i in range(warmup_len):
            x = self.forward(x, u[i])

            prediction_w[i] = self.model_r.predict(x.to(self.device))

        #Prediction
        for i in range(pred_len-1):
            x = self.forward(x, prediction_w[warmup_len + i-1])

            prediction_w[warmup_len + i] = self.model_r.predict(x.to(self.device))

        prediction_c = torch.zeros(pred_len, 1)

        x0 = torch.from_numpy(self.pca.inverse_transform(self.scaler.inverse_transform(self.model_h.predict(u[warmup_len-k:warmup_len].T.to(self.device)).detach().cpu().numpy()))).float().reshape(-1).to(device=self.device)
        
        x = x0 + torch.rand(x0.shape, device=self.device)*err

        
        prediction_c[0] = self.model_r.predict(x.to(self.device))
        for i in range(pred_len-1):
            x = self.forward(x, prediction_c[i])

            prediction_c[i+1] = self.model_r.predict(x.to(self.device))
        
        return prediction_c, prediction_w


spectral_radius = 0.9
leaky_rate = 0.7
train_len = 5000
discard = 1000
pred_len = 200
warmup_len = 7
k=7
def train_esn():
    scaler = MinMaxScaler(feature_range=(-0.2,0.2))
    esn = ESN(900, spectral_radius, leaky_rate)
    train_data = torch.from_numpy(scaler.fit_transform(np.loadtxt('train_data/Lorenz_Train.txt').reshape(-1,1))).float() 
    x_0 = esn.fit(train_data, train_len, discard, k)
    test_data = torch.from_numpy(scaler.transform(np.loadtxt('train_data/Lorenz_Test.txt')[:10000].reshape(-1,1))).float()
    prediction_c, prediction_w = esn.predict(test_data, pred_len, warmup_len, k)
    return test_data, prediction_c, prediction_w, esn

def plot_cold_warm(test_data, prediction_w, prediction_c, prediction_c_e, name):
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['axes.titley'] = 1.0    
    plt.rcParams['axes.titlepad'] = -10  
    fig = plt.figure(figsize=(10, 4))
    ax = fig.subplots()
    time = np.linspace(21, 25, 200)
    time2 = np.linspace(21-0.02*7, 21, 7)
    
    ax.plot(time2, test_data[2:9].detach().cpu().numpy(), c='red', lw = 1)
    ax.plot(time, test_data[8:pred_len+warmup_len+1].detach().cpu().numpy(), label='Actual', c='red', lw = 1)
    ax.plot(time, prediction_c.detach().cpu().numpy(), label='Cold', c='blue', lw = 1)
    ax.plot(time2, prediction_w[:7].detach().cpu().numpy(), label='Warmup', c='green', lw = 1)
    ax.plot(time, prediction_w[6:-1].detach().cpu().numpy(), label='Warm', c='orange', lw = 1)
    ax.legend(fontsize = 11, loc = 'upper right')   

    plt.savefig(f'Img/Lorenz_{name}.png')
    np.save(f'pred_data/Lorenz_{name}.npy', np.vstack((test_data[1:pred_len+warmup_len+1].detach().cpu().numpy(), prediction_c.detach().cpu().numpy(), prediction_w.detach().cpu().numpy())).T)

    
def predict_with_error(test_data, esn, err):
    prediction_c, prediction_w = esn.predict(test_data, pred_len, warmup_len, k, err)
    return prediction_c, prediction_w

if input('Train ESN? (y/n)') == 'y':
    cont = 'y'
    i = 56
    while True:
        torch.manual_seed(i)
        test_data, prediction_c, prediction_w, esn = train_esn()
        prediction_c_e, prediction_w = predict_with_error(test_data, esn, 0.03)
        plot_cold_warm(test_data, prediction_w, prediction_c, prediction_c_e, i)
        cont = input('Continue? (y/n)')
        i += 1
        if cont == 'n':
            continue
        err = np.concatenate([10*list(np.random.uniform(0, 0.03, 1000))])
        
        pos = 0
        mse = np.zeros(err.shape)
        prediction_c, prediction_w = predict_with_error(test_data, esn, 0.03)
        mse[0] = mean_squared_error(test_data[warmup_len+1:warmup_len+101].detach().cpu().numpy(), prediction_c[:100].detach().cpu().numpy())
        for i in range(1,len(err)):
            if i%1000 == 0:
                pos+= 100
            prediction_c, prediction_w = predict_with_error(test_data[pos:], esn, err[i])
            mse[i] = mean_squared_error(test_data[warmup_len+1+pos:warmup_len+101+pos].detach().cpu().numpy(), prediction_c[:100].detach().cpu().numpy())
            print(f"{i}/{len(err)}")

        np.savetxt(f'pred_data/Lorenz_Error_MSE_{i}.txt', mse)
        np.savetxt(f'pred_data/Lorenz_Error_{i}.txt', err)
        break
