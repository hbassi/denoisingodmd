import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
alphas = [0.001, 0.01, 0.1, 1.0, 10, 100]
betas = [0.001, 0.01, 0.1, 1.0, 10, 100]
gammas = [0.001, 0.01, 0.1, 1.0, 10, 100]

torch.manual_seed(999)

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
        x = self.fc5(x)

        return x

class ResidualLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.1):
        super(ResidualLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, predicted_eigval, predicted_eigvec):
        residual = torch.linalg.norm(tH @ predicted_eigvec - predicted_eigval * predicted_eigvec)**2 / torch.linalg.norm(tH)
        rayleigh_quotient = (predicted_eigvec.conj().T @ tH @ predicted_eigvec).real
        norm_penalty = self.alpha * rayleigh_quotient

        unit_norm_penalty = self.beta * (torch.linalg.norm(predicted_eigvec) - 1)**2
        eigenvalue_penalty = self.gamma * predicted_eigval

        # Total loss
        total_loss = residual + norm_penalty + unit_norm_penalty + eigenvalue_penalty

        return total_loss

size = 9  
input_size = 750 + 2 * size
hidden_size = 1024
output_size = 1 + 2 * size
num_epochs = 7500
learning_rate = 0.01
Tmax = 750
dt = 0.1
np.random.seed(999)
device = 'cuda'

Hr = np.random.randn(size**2).reshape((size, size))
Hi = np.random.randn(size**2).reshape((size, size))
Hraw = Hr + 1j*Hi
H = 0.5 * (Hraw + Hraw.conj().T) 
tH = torch.from_numpy(H).to(device)


print('T/F: H is Hermitian: ', (H == H.conj().T).all())
evals, evecs = np.linalg.eigh(H)


dot_product = np.dot(evecs.T.conj(), evecs)
identity = np.eye(evecs.shape[1])
tolerance = 1e-10

if np.allclose(dot_product, identity, atol=tolerance):
    print("Eigenvectors are orthogonal within the given tolerance.")
else:
    print("Eigenvectors are not orthogonal.")


phi0 = (np.random.randn(size) + 1j * np.random.randn(size)) 

#compute observables for data s(k\Delta t) = \langle \phi_0 | e^-iHk\Delta t | \phi_0 \rangle
dataS = np.zeros(Tmax, dtype=np.complex128)
for k in range(Tmax):
    dataS[k] = phi0.conj().T @ scipy.linalg.expm(-1j* H * k * dt) @ phi0


phi0HF = (np.random.randn(size) + 1j * np.random.randn(size)) 

input_data = np.concatenate((dataS.real, phi0HF.real,phi0HF.imag))
input_data = input_data.reshape((1,input_data.shape[0]))
input_data = torch.Tensor(input_data).double().to(device)

for alpha in alphas:
    for beta in betas:
        for gamma in gammas:
            print(f'ALPHA: {alpha}, BETA: {beta}, GAMMA: {gamma}')
            
            model = FullyConnectedNet(input_size, hidden_size, output_size).to(device)
            criterion = ResidualLoss(alpha=alpha, beta=beta, gamma=gamma)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)


            for epoch in trange(num_epochs):
                model.train()

                outputs = model(input_data)
                pred_eval = outputs[:,0]
                pred_evec_all = outputs[:,1:].T
                complex_evec_all = torch.complex(pred_evec_all[:9], pred_evec_all[9:])
                loss = criterion(pred_eval, complex_evec_all)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch+1) % 1000 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')

            print("Training complete")

            model.eval()
            with torch.no_grad():  
                prediction = model(input_data)

            pred_eval = prediction[0,0].item()
            pred_evec = torch.complex(prediction[0, 1:10], prediction[0, 10:])


            eig_val_idx_errs = []
            for i in range(evals.shape[0]):
                eig_val_idx_errs.append(abs(evals[i] - pred_eval))


            # plt.semilogy(range(0, len(evals)), eig_val_idx_errs)
            # plt.xlabel('Eigenvalue index number')
            # plt.ylabel('MAE')
            # plt.title('FFNN Eigenvalue prediction MAE');
            # plt.tight_layout()

            closest_index = np.argmin(np.abs(evals - pred_eval))
            # plt.scatter(range(len(evals)), evals, label='Eigenvalues')
            # plt.scatter([closest_index], [pred_eval], color='red', label='Predicted Eigenvalue')
            # plt.xlabel('Index')
            # plt.ylabel('Eigenvalue')
            # plt.title('Eigenvalues Scatter Plot')
            # plt.legend()
            # plt.tight_layout()

            print('Closest index: ', closest_index)
            print('MAE: ', eig_val_idx_errs[closest_index] )
           
                
            true_evec = evecs[:, closest_index]
            true_evec = torch.from_numpy(true_evec).to(pred_evec.device)
            pred_evec_norm = pred_evec / torch.linalg.norm(pred_evec)

            cosine_similarity = torch.abs(torch.dot(pred_evec_norm.conj(), true_evec))
            print(f'Cosine similarity between predicted and true eigenvector: {cosine_similarity.item()}')
            angle = torch.acos(cosine_similarity).item()
            print(f'Angle between predicted and true eigenvector (radians): {angle}')
            with open('training_logs.txt', 'a') as f:
                f.write(f'Closest predicted index: {closest_index}, MAE: {eig_val_idx_errs[closest_index]}, Loss: {loss.item()}, Alpha: {alpha}, Beta: {beta}, Gamma: {gamma}, Cosine similarity: {cosine_similarity}\n')
            print('=====================================================================================================')
