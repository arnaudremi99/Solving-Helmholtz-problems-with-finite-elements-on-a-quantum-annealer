# General
#---
import numpy as np #
import math
import matplotlib.pyplot as plt #
from IPython.display import set_matplotlib_formats
import xlrd
from functools import partial
from scipy.optimize import curve_fit #
from scipy.interpolate import interp1d
from scipy.linalg import eig
import itertools
from itertools import product
from collections import defaultdict
import random
from tqdm import tqdm

# D-Wave
#---
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler
import dimod
import neal





class Solver_parameters:
    def __init__(self, solver_type='QA', maxiter=50, maxiter_gamma=10, annealingtime=None, nsteps=None, numreads=10, N_iter_per_reads=None, alpha=0.5, linesearch_method='bisection', perturbation=0):
        self.solver_type       = solver_type
        self.numreads          = numreads
        self.maxiter           = maxiter
        self.maxiter_gamma     = maxiter_gamma
        self.linesearch_method = linesearch_method
        self.alpha             = alpha
        self.perturbation      = perturbation  # For simulating Integrated Control Errors (ICE) in quantum hardwares
        # SQA
        self.nsteps            = nsteps
        self.annealingtime     = annealingtime #in microseconds
        # SA
        self.N_iter_per_reads  = N_iter_per_reads
        

class Data_problem:
    def __init__(self, H, M, N, K, n=0):
        self.M = M
        self.N = N
        self.K = K
        self.n = n
        self.H = self.proj_H(H)

    def proj_H(self, H):
        eigenvalues, eigenvectors = eig(H, self.M)
        idx          = eigenvalues.argsort()[::1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        H_proj = H
        for n in range(self.n):
            phi_n     = eigenvectors[:,n]
            lambda_n  = eigenvalues[n]
            proj_n = get_projector(phi_n, self.M)
            beta_n = 10*np.trace(H)/np.trace(self.M)
            H_proj += beta_n * proj_n

        return H_proj
    
class Data_solution:
    def __init__(self, u, dz, gamma=0):
        self.u     = u
        self.u_updt= u
        self.dz    = dz
        self.gamma = gamma
        self.gamma_store = []
        self.cost      = []
        self.u_store   = []
        self.gamma_bounds  = [None,None]
        self.energy_bounds = [None,None] # for secant method
        self.psi   = None

class Solver():
    def __init__(self, data_solution, data_problem, solver_parameters):
        self.data_solution     = data_solution
        self.data_problem      = data_problem
        self.solver_parameters = solver_parameters

    def solve(self):
        if self.solver_parameters.solver_type == 'QA':
            u = self.solve_QA()
        elif self.solver_parameters.solver_type == 'SA':
            u = self.solve_SA()
        elif self.solver_parameters.solver_type == 'customSA':
            u = self.solve_customSA()
        else:
            print('error : unknown solver type')
        return u

    def init_gamma(self):
        Q_H     = assemble_Q(self.data_problem.N, self.data_problem.K, self.data_problem.H, self.data_solution.u, self.data_solution.dz)
        Q_M     = assemble_Q(self.data_problem.N, self.data_problem.K, self.data_problem.M, self.data_solution.u, self.data_solution.dz)
        factor  = 1
        if self.solver_parameters.linesearch_method == 'bisection':
            gamma_min =  self.data_solution.gamma - factor*abs(Q_H).max()
            gamma_max =  self.data_solution.gamma + factor*abs(Q_H).max()
            iteration = 0
            while True:
                self.data_solution.gamma = gamma_min
                u_min = self.solve()
                E_min = u_min@(self.data_problem.H-gamma_min*self.data_problem.M)@u_min
                self.data_solution.gamma = gamma_max
                u_max = self.solve()
                E_max = u_max@(self.data_problem.H-gamma_max*self.data_problem.M)@u_max
                self.data_solution.gamma = (gamma_min + gamma_max)/2
                # Stopping criterion
                if E_min >= 0 and E_max < 0:
                    break
                factor *= 2
                if E_min < 0:
                    gamma_min  =  self.data_solution.gamma - factor*abs(Q_H).max()
                if E_max >= 0:
                    gamma_max  =  self.data_solution.gamma + factor*abs(Q_H).max()
        
                iteration += 1
                if iteration > 10:
                    print('Bounds cannot be found !')
                    self.data_solution.gamma_bounds = [None, None]
                    return 
            self.data_solution.gamma_bounds = [gamma_min, gamma_max]            
        else:
            print('error : unknown linesearch method')

    def update_gamma(self):
        if self.solver_parameters.linesearch_method == 'bisection':
            print(f'value = {self.data_solution.u_updt@(self.data_problem.H-self.data_solution.gamma*self.data_problem.M)@self.data_solution.u_updt}')
            if self.data_solution.u_updt@(self.data_problem.H-self.data_solution.gamma*self.data_problem.M)@self.data_solution.u_updt >= 0:
                self.data_solution.gamma_bounds[0] = self.data_solution.gamma
            else:
                self.data_solution.gamma_bounds[1] = self.data_solution.gamma
            self.data_solution.gamma = 0.5 * (self.data_solution.gamma_bounds[0] + self.data_solution.gamma_bounds[1])
        else:
            print('error : unknown linesearch method')
        return 

    def update_dz(self):
        self.data_solution.dz *= self.solver_parameters.alpha

    def update_cost(self):
        self.data_solution.cost.append(np.linalg.norm(self.data_solution.u)**(-1) * np.linalg.norm(self.data_problem.H@self.data_solution.u - self.data_solution.gamma*self.data_problem.M@self.data_solution.u))
        self.data_solution.gamma_store.append(self.data_solution.gamma)

    def solve_QA(self):
        # Checks annealing type
        if self.solver_parameters.annealingtime < 0.5:
            print(f'annealing time = {1000*self.solver_parameters.annealingtime} ns < 500 ns, fast annealing protocol is selected')
            fast_anneal=True
        else:
            fast_anneal=False

        def logical_solution(sample_set,Q,N,K):
            q = list( sample_set.first.sample.values() )
            k = 0
            Q_mapping = defaultdict(int)  # keys are the logical variables, values are the index at which they are stored
            for vars in list(sample_set.variables):
                if vars in range(0,N*K):
                    Q_mapping[vars] += k
                k+=1
            q_logical = np.zeros([N*K])
            for i in range(N*K):
                q_logical[i] = q[Q_mapping[(i)]]
            return q_logical

        
        Q = self.get_qubo()

        noise = True
        if noise==True:
            maxval = max([abs(Q.max()), abs(Q.min())])
            for i in range(len(Q)):
                for j in range(len(Q)):
                    Q[i,j] += (np.random.normal(0, self.solver_parameters.perturbation)) * maxval


        Q_dict   = self.matrix_to_dict(Q)
        sampler  = EmbeddingComposite(DWaveSampler( compress_qpu_problem_data=False ))
        S        = sampler.sample_qubo(Q_dict, num_reads=self.solver_parameters.numreads, annealing_time=self.solver_parameters.annealingtime,  fast_anneal=fast_anneal, label='aqae')
        q        = logical_solution(S,Q_dict,self.data_problem.N,self.data_problem.K)
        v        = to_decimal(q, self.data_problem.N, self.data_problem.K, self.data_solution.u, self.data_solution.dz)
        return v

    def solve_SA(self):
        # Checks annealing type
        if self.solver_parameters.annealingtime < 0.5:
            print(f'annealing time = {1000*self.solver_parameters.annealingtime} ns < 500 ns, fast annealing protocol is selected')
            fast_anneal=True
        else:
            fast_anneal=False

        def logical_solution(sample_set,Q,N,K):
            q = list( sample_set.first.sample.values() )
            k = 0
            Q_mapping = defaultdict(int)  # keys are the logical variables, values are the index at which they are stored
            for vars in list(sample_set.variables):
                if vars in range(0,N*K):
                    Q_mapping[vars] += k
                k+=1
            q_logical = np.zeros([N*K])
            for i in range(N*K):
                q_logical[i] = q[Q_mapping[(i)]]
            return q_logical
        

        Q = self.get_qubo()

        noise = True
        if noise==True:
            maxval = max([abs(Q.max()), abs(Q.min())])
            for i in range(len(Q)):
                for j in range(len(Q)):
                    Q[i,j] += (np.random.normal(0, self.solver_parameters.perturbation)) * maxval


        Q_dict   = self.matrix_to_dict(Q)
        sampler = neal.SimulatedAnnealingSampler()

        S        = sampler.sample_qubo(Q_dict, num_reads=self.solver_parameters.numreads, annealing_time=self.solver_parameters.annealingtime, fast_anneal=fast_anneal, label='IQAE')
        q        = logical_solution(S,Q_dict,self.data_problem.N,self.data_problem.K)
        v        = to_decimal(q, self.data_problem.N, self.data_problem.K, self.data_solution.u, self.data_solution.dz)
        return v


    def solve_customSA(self):
        
        def update(s):
            i = np.random.randint(len(s))
            
            s_proposal = np.copy(s)
            s_proposal[i] *= -1
    
            return s_proposal
        
        def objective_function(J,h,s):
            Energy = s@J@s + h@s
            return Energy

        def simulated_annealing(J, h, N_spins, n_iter_per_reads, num_reads):
            s   = np.ones([num_reads, N_spins])
            for read in range(num_reads):
                s[read,:] = np.ones(N_spins) # not random for "better" reproductibility
                
            E   = np.zeros(num_reads)
            E_iter = np.zeros(n_iter_per_reads)
            p   = 1                     # geometric scheduling parameter
            T_i =  2*abs(s[0,:]@J@s[0,:] + h@s[0,:])   # Initial temperature
            T_f = T_i/100                              # Final   temperature
            t   = np.linspace(0, (T_i/T_f - 1)**(1/p) , n_iter_per_reads)
            T   = T_i/(t**p + 1)
            
            for read in range(num_reads):
                E[read] = objective_function(J,h,s[read,:])
                
                for i in range(n_iter_per_reads):
                    s_proposal = update(s[read,:])
                    E_proposal = objective_function(J,h,s_proposal)
                    dE = E_proposal - E[read]
                    
                    if np.exp(-dE/T[i]) > random.random():
                        s[read,:] = s_proposal
                        E[read] = objective_function(J,h,s[read,:])
                    E_iter[i] = objective_function(J,h,s[read,:])
                if False:
                    plt.figure(figsize=(4, 3))
                    plt.plot(t,E_iter, linestyle = '-', linewidth=2, color='blue')
                    plt.plot(t,T, linestyle = '-', linewidth=2, color='red')
                    plt.xlabel(r'$t$')
                    plt.ylabel(r'Energy')
                    plt.legend([r'$E$', r'$k_BT$'])
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.show()
        
            opt_index = np.argmin(E)
            E_opt = E[opt_index]
            s_opt = s[opt_index,:]
        
            return s_opt, E_opt

        J,h = self.get_ising()
        s, E = simulated_annealing(J,h,self.data_problem.N * self.data_problem.K, self.solver_parameters.N_iter_per_reads, self.solver_parameters.numreads)
        q    = 0.5 * (s+1)
        v    = to_decimal(q, self.data_problem.N, self.data_problem.K, self.data_solution.u, self.data_solution.dz)   

        return v

    def get_qubo(self):
        Q_H = assemble_Q(self.data_problem.N, self.data_problem.K, self.data_problem.H, self.data_solution.u, self.data_solution.dz)
        Q_M = assemble_Q(self.data_problem.N, self.data_problem.K, self.data_problem.M, self.data_solution.u, self.data_solution.dz)
        Q = Q_H - self.data_solution.gamma * Q_M
        return Q

    def get_ising(self):
        Q = self.get_qubo()
        J,h = self.qubo_to_ising(Q)
        return J,h

    def qubo_to_ising(self,Q):
        n = len(Q)
        J = np.zeros((n, n))
        h = np.zeros(n)
    
        for i in range(n):
            h[i] = (sum(Q[:, i]) + sum(Q[i,:]))/4
    
        for i in range(n):
            for j in range(i+1, n):
                J[i, j] = Q[i, j]/4
                J[j, i] = Q[j, i]/4
                
        maxval   = max([abs(J.max()), abs(J.min()), abs(h.max()), abs(h.min())])
        J       /= maxval
        h       /= maxval

        return J, h

    def matrix_to_dict(self,mat):
        Q = defaultdict(float)
        num_rows, num_columns = mat.shape
        for i in range(num_rows):
            for j in range(num_columns):
                if mat[i,j] != 0:
                    Q[(i,j)] += mat[i,j]
        return Q


# Q matrix assembly
#---
def assemble_Q(N, K, M, v0 = None, dz = None):
    if v0 is None:
        v0 = np.zeros(N)
    if dz is None:
        dz = 2 * np.ones(N)
    eta = v0 - dz/2 + 2**(-K) * dz/2
    matrix_Q = np.zeros([N*K, N*K])
    for i in range(N):
        bound_i = [v0[i] - dz[i]/2, v0[i] + dz[i]/2]
        for j in range(N):
            bound_j = [v0[j] - dz[j]/2, v0[j] + dz[j]/2]
            bloc    = binarization_bloc(K, bound_i, bound_j)
            matrix_Q[K*i:K*(i+1), K*j:K*(j+1)] += (M[i,j])*bloc
            if i == j:
                bloc_diag = np.sqrt(np.diag(np.diag(bloc)))
                matrix_Q[K*i:K*(i+1), K*j:K*(j+1)] += (M[i,:] @ eta)*bloc_diag
                matrix_Q[K*i:K*(i+1), K*j:K*(j+1)] += (eta @ M[:,i])*bloc_diag
    return matrix_Q


# binarization
#---
def binarization_bloc(K, bound_k=[-1,1], bound_l=[-1,1]):
    bloc = np.ones([K,K])
    for k in range(K):
        bloc[k,:] *= (bound_k[1]-bound_k[0]) * 2**(k-K)
    for l in range(K):
        bloc[:,l] *= (bound_l[1]-bound_l[0]) * 2**(l-K)
    return bloc


# binary to decimal array converter
#---
def to_decimal(q, N, K, v0=None, dz=None):
    if v0 is None:
        v0 = np.zeros(N)
    if dz is None:
        dz = 2 * np.ones(N)
    eta = v0 - dz/2 + 2**(-K) * dz/2
    v = eta
    for i in range(N):
        for k in range(K):# q_i    *       b_i
            v[i] +=       q[K*i+k] * dz[i] * 2**(k-K)
    return v


def get_probabilities(psi):
    P = np.zeros(len(psi))
    for k in range(len(psi)):
        P[k] = np.abs(psi[k].squeeze())**2
    return P


def get_minimum_energy_config(qq, Q):
    numreads, N_spins = qq.shape
    E = np.zeros(numreads)
    for k in range(numreads):
        E[k] = qq[k,:] @ Q @ qq[k,:]
    min_ind = np.argmin(E)
    return qq[min_ind,:]


def get_projector(phi_1, B):
    """
    Computes the projector P = B * phi_1 * phi_1^T * B
    for a generalized eigenvalue problem.

    Parameters:
    - phi_1: numpy array, the first eigenvector (1D array)
    - B: numpy array, the matrix B in the generalized eigenvalue problem

    Returns:
    - P: numpy array, the projector matrix
    """
    # Check the shape of phi_1 before reshaping
    if phi_1.size == 0:
        raise ValueError("phi_1 is empty!")
    
    # Ensure phi_1 is a 1D array with the right shape
    if phi_1.ndim != 1:
        raise ValueError(f"phi_1 should be a 1D array, but has shape {phi_1.shape}")
    
    # Reshape phi_1 to a column vector (shape (n, 1))
    phi_1 = phi_1.reshape(-1, 1)
    
    # Check if B has the correct shape
    if B.shape[0] != B.shape[1]:
        raise ValueError("Matrix B should be square")
    if B.shape[0] != phi_1.shape[0]:
        raise ValueError(f"The shape of B ({B.shape}) and phi_1 ({phi_1.shape}) do not align")
    
    # Compute B * phi_1
    B_phi_1 = np.dot(B, phi_1)
    
    # Compute the outer product phi_1^T * B_phi_1
    phi_1_B_phi_1_T = np.dot(phi_1.T, B_phi_1)
    
    # Finally compute the projector P = B * phi_1 * phi_1^T * B
    P = np.dot(B_phi_1, B_phi_1.T) / phi_1_B_phi_1_T
    
    return P




def save_results(iqae):
    filename_cost = f"results/cost_N={iqae.data_problem.N}_K={iqae.data_problem.K}_maxiter={iqae.solver_parameters.maxiter}_maxiter_gamma={iqae.solver_parameters.maxiter_gamma}_numreads={iqae.solver_parameters.numreads}_alpha={iqae.solver_parameters.alpha}_n={iqae.data_problem.n}_annealingtime={iqae.solver_parameters.annealingtime}_solvertype={iqae.solver_parameters.solver_type}_N_flips={iqae.solver_parameters.N_iter_per_reads}_perturbation={iqae.solver_parameters.perturbation}.txt"
    with open(filename_cost, 'a') as file:
        np.savetxt(file, [iqae.data_solution.cost], delimiter=' ', newline='\n', comments='')
    filename_v = f"results/u_N={iqae.data_problem.N}_K={iqae.data_problem.K}_maxiter={iqae.solver_parameters.maxiter}_maxiter_gamma={iqae.solver_parameters.maxiter_gamma}_numreads={iqae.solver_parameters.numreads}_alpha={iqae.solver_parameters.alpha}_n={iqae.data_problem.n}_annealingtime={iqae.solver_parameters.annealingtime}_solvertype={iqae.solver_parameters.solver_type}_N_flips={iqae.solver_parameters.N_iter_per_reads}_perturbation={iqae.solver_parameters.perturbation}.txt"
    with open(filename_v, 'a') as file:
        np.savetxt(file, [iqae.data_solution.u], delimiter=' ', newline='\n', comments='')
    filename_gamma = f"results/lambda_N={iqae.data_problem.N}_K={iqae.data_problem.K}_maxiter={iqae.solver_parameters.maxiter}_maxiter_gamma={iqae.solver_parameters.maxiter_gamma}_numreads={iqae.solver_parameters.numreads}_alpha={iqae.solver_parameters.alpha}_n={iqae.data_problem.n}_annealingtime={iqae.solver_parameters.annealingtime}_solvertype={iqae.solver_parameters.solver_type}_N_flips={iqae.solver_parameters.N_iter_per_reads}_perturbation={iqae.solver_parameters.perturbation}.txt"
    with open(filename_gamma, 'a') as file:
        np.savetxt(file, [iqae.data_solution.gamma_store], delimiter=' ', newline='\n', comments='')
    filename_u_store = f"results/phi_store_N={iqae.data_problem.N}_K={iqae.data_problem.K}_maxiter={iqae.solver_parameters.maxiter}_maxiter_gamma={iqae.solver_parameters.maxiter_gamma}_numreads={iqae.solver_parameters.numreads}_alpha={iqae.solver_parameters.alpha}_n={iqae.data_problem.n}_annealingtime={iqae.solver_parameters.annealingtime}_solvertype={iqae.solver_parameters.solver_type}_N_flips={iqae.solver_parameters.N_iter_per_reads}_perturbation={iqae.solver_parameters.perturbation}.txt"
    with open(filename_u_store, 'a') as file:
        np.savetxt(file, iqae.data_solution.u_store, delimiter=' ', newline='\n', comments='')



