import numpy as np
import scipy
from scipy import io
from scipy.linalg import svd, hankel, eig
from matplotlib import pyplot as plt
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.optim.lr_scheduler import CosineAnnealingLR

alphas = [1.0]
#betas = [0.1, 1.0, 2.0, 5.0, 10.0]
betas = alphas  #[ 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
#gammas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 10.0]

class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size).double()
        self.fc2 = nn.Linear(hidden_size, hidden_size).double()
        self.fc3 = nn.Linear(hidden_size, hidden_size).double()
        self.fc4 = nn.Linear(hidden_size, hidden_size).double()
        self.fc5 = nn.Linear(hidden_size, output_size).double()
    
        self.activation = nn.Tanh()
    
        self._initialize_weights()

    def _initialize_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.fc4.weight)
        init.xavier_uniform_(self.fc5.weight)
        
        
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)
        if self.fc3.bias is not None:
            init.zeros_(self.fc3.bias)
        if self.fc4.bias is not None:
            init.zeros_(self.fc4.bias)
        if self.fc5.bias is not None:
            init.zeros_(self.fc5.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.fc5(x)
        x_eval = x[:, 0].unsqueeze(1)
        x_evec = x[:, 1:]
        x_evec = - x_evec / torch.linalg.norm(x_evec)
        x = torch.concat((x_eval, x_evec), axis = 1)
        return x

molecule = 'Cr2'
noise = 1.0
overlap = 0.0# 0: HF
Tmax = 600
step = 1
dt = 3

device = 'cuda'
size = 4000
input_size = Tmax +  size
hidden_size = 1024
output_size = 1 + size
num_epochs = 10000
learning_rate = 0.1

def generate_samples(E, psi0, dt=1, nb=100):
    H = np.zeros(nb, dtype=np.complex128)
    S = np.zeros(nb, dtype=np.complex128)
    for j in trange(nb):
        H[j] = np.sum(E * np.abs(psi0)**2 * np.exp(-1j * E * j * dt))
        S[j] = np.sum(np.abs(psi0)**2 * np.exp(-1j * E * j * dt))
    return H, S

def lam2lamt(lam, lammin, lammax):
    lamt = np.pi / 2 * (lam - (lammin + lammax) / 2) / (lammax - lammin)
    return lamt

def lamt2lam(lamt, lammin, lammax):
    lam = lamt * 2 / np.pi * (lammax - lammin) + (lammin + lammax) / 2
    return lam

def make_hankel(data, m, n):
    return hankel(data[:m], data[m-1:m+n-1])

def generate_phi(overlap, N):
    phi = np.zeros((N,1))
    phi[0] = np.sqrt(overlap)
    phi[1:] = np.sqrt((1 - phi[0]**2) / (N - 1))
    return phi

seed = 100
np.random.seed(seed)

data_cr2 = scipy.io.loadmat('./data/Cr2_4000.mat')
psiHF = data_cr2['psiHF']
E = data_cr2['E']
Et = lam2lamt(E,E[0],E[-1])

if overlap == 0:
    dataH,dataS = generate_samples(Et,psiHF,dt,Tmax)
    phi = psiHF
else:
    phi = generate_phi(overlap,len(Et));
    print('generated phi')
    [dataH,dataS] = generate_samples(Et,phi,dt,Tmax);

dataH = (dataH + noise * np.random.randn(Tmax) + 1j * noise * np.random.randn(Tmax)).real
dataS = (dataS + noise * np.random.randn(Tmax) + 1j * noise * np.random.randn(Tmax)).real
# (num_samples, input_size)
dataS = dataS.reshape((1,Tmax))

class ResidualLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1):
        super(ResidualLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, predicted_eigval, predicted_eigvec, tH, lower_bound, upper_bound):
        tH = tH.detach()
        # lower_bound = lower_bound.detach()
        # upper_bound = upper_bound.detach()

        residual = torch.linalg.norm(tH @ predicted_eigvec - predicted_eigval * predicted_eigvec) 
        
        # Lower bound penalty (one-sided)
        lower_bound_violations = (predicted_eigval < lower_bound).float()
        lower_bound_penalty = self.alpha * (lower_bound_violations * (lower_bound - predicted_eigval) ** 4) 
        
        # Upper bound penalty (one-sided)
        upper_bound_violations = (predicted_eigval > upper_bound).float()
        upper_bound_penalty = self.beta * (upper_bound_violations * (predicted_eigval - upper_bound) ** 4)

        # Total loss
        total_loss = residual + lower_bound_penalty + upper_bound_penalty

        return total_loss

    
    
class MAEResidual(nn.Module):
    def __init__(self):
        super(MAEResidual, self).__init__()
   
     
    def forward(self, predicted_eigval, estimate_eigvec):
        total_loss = torch.linalg.norm(tH @ estimate_eigvec - predicted_eigval * estimate_eigvec)  
        return total_loss

def find_perpendicular_vector(v):
    v = np.array(v)
    dim = v.shape[0]
    random_vector = np.random.rand(dim)
    perpendicular_vector = None
    while np.allclose(random_vector, v):
        random_vector = np.random.rand(dim)
    random_vector -= random_vector.dot(v) * v / np.linalg.norm(v)**2
    perpendicular_vector = random_vector
    return perpendicular_vector / np.linalg.norm(perpendicular_vector)
res = find_perpendicular_vector(np.squeeze(psiHF, 1))

H = scipy.io.loadmat('./data/H4000.mat')['H']
tH = torch.from_numpy(H).to(device)
dataS = torch.tensor(dataS).double()
psiHF = torch.tensor(psiHF).double()
phi = torch.tensor(phi).double()
ref_state = psiHF
input_data = torch.concat((dataS, ref_state.T), axis = 1).to(device)

E0 = (psiHF.conj().T @ H @ psiHF).item()
E1 = res.conj().T @ H @ res
sigma = (np.sqrt( psiHF.conj().T @ (H @ H) @ psiHF - (psiHF.conj().T @ H @ psiHF) ** 2)).item()
upper_bound = 0.5 * (E0 + E1 - np.sqrt((E1 - E0) ** 2 + 4 * sigma ** 2) )
lower_bound = E0 - (sigma ** 2 / (E1 - E0))

#print('Pollack estimate: ', estimate_GSE)
#print('Improved bound: ', abs(estimate_GSE - E[0]))
true_evec = scipy.linalg.null_space(H - E[0] * np.eye(H.shape[0]))
print(E[0] - (true_evec.conj().T @ H @ true_evec)/np.linalg.norm(true_evec))
true_evec = torch.from_numpy(true_evec).to(device)
for alpha in alphas:
    beta = alpha
    #print(f'ALPHA: {alpha}, BETA: {beta}, GAMMA: {gamma}')
    print(f'ALPHA: {alpha}, BETA: {beta}')
    torch.manual_seed(999)
    model = FullyConnectedNet(input_size, hidden_size, output_size).to(device)
    criterion1 = ResidualLoss(alpha=alpha, beta=beta)
    criterion2 = MAEResidual()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    val_preds = []
    val_preds_vec = []
    val_mae_gse = []
    val_mae_ci = []
    residuals = []
    angles = []
    vec_residuals = []

    for epoch in trange(num_epochs):
        model.train()

        outputs = model(input_data)
        pred_eval = outputs[:, 0]
        pred_evec_all = outputs[:, 1:].T
        # if epoch > 5000:
        #     loss = criterion2(pred_eval, pred_evec_all)
        # else: 
        #     loss = criterion1(pred_eval, pred_evec_all, tH, lower_bound, upper_bound)
        #     estimate_eigvec = pred_evec_all
        loss = criterion1(pred_eval, pred_evec_all, tH, lower_bound, upper_bound)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch+1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
            model.eval()
            
            with torch.no_grad():  
                #prediction = model(-torch.ones(testdata.shape[1]).reshape((1,testdata.shape[1])).double().to(device))
                prediction = model(input_data)
            pred_eval = prediction[0,0].item()
            pred_evec = prediction[0, 1:].reshape((size, 1))
            residual = torch.linalg.norm(tH @ pred_evec - pred_eval * pred_evec).item() 
            vec_residual = torch.linalg.norm(pred_evec - true_evec)
            
            
            eig_val_idx_errs = []
            for i in range(E.shape[0]):
                eig_val_idx_errs.append(abs(E[i] - pred_eval))
            closest_index = np.argmin(np.abs(E - pred_eval))
            
            mae_closest_index = abs(E[closest_index] - pred_eval)
            mae_gse = abs(E[0] - pred_eval)
            mae_ub = abs(upper_bound - pred_eval)
            mae_lb = abs(lower_bound - pred_eval)
            
            cosine_similarity = (torch.dot(pred_evec.conj().squeeze(1), true_evec.squeeze(1)))
            angle = torch.acos(cosine_similarity).item() * 180/np.pi
            
            residuals.append(residual)
            angles.append(angle)
            val_preds.append(pred_eval)
            val_mae_gse.append(mae_gse.item())
            val_preds_vec.append(pred_evec.cpu().numpy())
            vec_residuals.append(vec_residual.item())

            print(f'Prediction: {pred_eval}, Closest index: {closest_index}, MAE_CI: {mae_closest_index}, MAE_GSE: {mae_gse}, MAE_UB: {mae_ub}, MAE_LB: {mae_lb}')


    print("Training complete")

    model.eval()
    testdata = torch.Tensor(np.concatenate(((dataS + 2.0 * np.random.randn(dataS.shape[0])).real, psiHF.T), axis = 1)).reshape((1, Tmax + psiHF.shape[0])).double().to(device)
    with torch.no_grad():  
        #prediction = model(-torch.ones(testdata.shape[1]).reshape((1,testdata.shape[1])).double().to(device))
        prediction = model(testdata)
        
    pred_eval = prediction[0,0].item()
    pred_evec = prediction[0, 1:].reshape((size, 1))
    
    cosine_similarity = (torch.dot(pred_evec.conj().squeeze(1), true_evec.squeeze(1)))
    print(f'Cosine similarity between predicted and true eigenvector: {cosine_similarity.item()}')
    angle = torch.acos(cosine_similarity).item()
    print(f'Angle between predicted and true eigenvector (radians): {angle}')
    residual = torch.linalg.norm(tH @ pred_evec - pred_eval * pred_evec) 
    eig_val_idx_errs = []
    for i in range(E.shape[0]):
        eig_val_idx_errs.append(abs(E[i] - pred_eval))
    closest_index = np.argmin(np.abs(E - pred_eval))
    print('Closest index: ', closest_index)
    print('MAE: ', eig_val_idx_errs[closest_index] )
    print('Residual: ', residual.item())
    
    with open('./logs/cr2_gse_one_sided_penalty_residual_loss_alpha='+str(alpha)+'_beta='+str(beta)+'_Tmax='+str(Tmax)+'_training_logs.txt', 'a') as f:
        f.write(f'Closest predicted index: {closest_index}, MAE: {eig_val_idx_errs[closest_index]}, Loss: {loss.item()}, Alpha: {alpha}, Beta: {beta}, Residual: {residual.item()}, Angle: {angle}\n')
        
    np.savez('./saved_data/Tmax='+str(Tmax)+'/cr2_gse_one_sided_penalty_residual_loss_alpha='+str(alpha)+'_beta='+str(beta)+'_Tmax='+str(Tmax)+'_prediction.npz', MAE=eig_val_idx_errs[closest_index], residual=residual.item(), angle=angle, pred_eval=pred_eval, pred_evec=pred_evec.cpu())
    #import pdb; pdb.set_trace()
    np.savez('./saved_data/Tmax='+str(Tmax)+'/cr2_gse_one_sided_penalty_residual_loss_alpha='+str(alpha)+'_beta='+str(beta)+'_Tmax='+str(Tmax)+'_training.npz', training_MAEs=val_mae_gse , training_residuals=residuals, training_angles=angles, training_preds=val_preds, training_vec_preds=val_preds_vec, training_vec_residuals=vec_residuals)
    torch.save(model.state_dict(), './saved_data/Tmax='+str(Tmax)+'/cr2_train_noise=1.0_HFgs_alpha_beta='+str(alpha)+'weights.pth')
    # print('=========================================================================================================================================================')


