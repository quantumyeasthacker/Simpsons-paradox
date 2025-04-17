import numpy as np
from scipy import optimize
import random

class Cell_Population:
    def __init__(self,
                    k_n0, # nutrient quality
                    ribo_div_noise, # noise due to unequal partitioning of ribosomes at division
                    f_i, # average flux allocation to unnecessary protein expression
                    tau_u, # unnecessary protein mean-reversion timescale
                    sigma_u, # volatility coefficient in unnecessary protein expression
                    tau_r, # ribosome mean-reversion timescale
                    sigma_r=0, # volatility coefficient in ribosome expression
                    num_cells_init=3000,
                    delta_t=0.05):
        self.init_conditions = []
        self.num_cells_init = num_cells_init
        self.delta_t = delta_t

        # parameters
        self.phiR_min = 0
        self.a_n = 1e-3 # level to trigger negative feeback inhibition of amino acid supply, Scott et al. 2014
        self.a_t = 1e-4 # amino acid level for efficient peptide elongation, Scott et al. 2014
        self.n_f = 2 # cooperativity in feedback
        self.n_g = 2
        # Kratz and Banerjee 2023
        self.alphaX = 4.5
        self.betaX = 1.1
        self.mu = 0.6
        self.kt0_mean = 5 # translational efficiency
        self.k_n0 = k_n0

        self.f_i = f_i
        self.phiRmax_mean = 0.55
        self.sigma_phiRmax = sigma_r
        self.sigma_u = sigma_u
        self.tau = tau_r
        self.tau_u = tau_u
        self.ribo_div_noise = ribo_div_noise


    # defining regulatory functions and their derivatives
    def f(self, a):
        return 1 / (1 + (a/self.a_n)**self.n_f) # regulatory function for k_n
    def f_prime(self, a):
        return -(self.n_f/self.a_n)*(a/self.a_n)**(self.n_f-1) / (1 + (a/self.a_n)**self.n_f)**2 # derivative of f w.r.t. a
    def g(self, a):
        return (a/self.a_t)**self.n_g / (1 + (a/self.a_t)**self.n_g) # regulatory function for k_t
    def g_prime(self, a):
        return (self.n_g/self.a_t)*(a/self.a_t)**(self.n_g-1) / (1 + (a/self.a_t)**self.n_g)**2 # derivative of g w.r.t. a

    # f_R, fraction of total cell synthesis capacity devoted to ribosome production
    def f_R(self, a, phiR_max, f_i):
        return (-self.f_prime(a)*self.g(a)*(phiR_max-f_i) + self.f(a)*self.g_prime(a)*self.phiR_min) / (-self.f_prime(a)*self.g(a) + self.f(a)*self.g_prime(a))
    # f_X, fraction of cell synthesis capacity devoted to division protein production
    def f_X(self, a, phiR_max, f_i):
        return self.alphaX*(phiR_max - self.f_R(a,phiR_max,f_i) - f_i) + self.betaX

    def GrowthRate(self, a, phi_R):
        # growth rate function
        k_t = self.k_t0 * self.g(a)
        k = k_t * (phi_R - self.phiR_min)
        return k

    def dphiR_dt(self, phi_R, t, a, phiR_max, f_i):
        # ribosome mass fraction ODE
        k_t = self.k_t0 * self.g(a) # translational efficiency
        dpdt = k_t * (phi_R - self.phiR_min) * (self.f_R(a,phiR_max,f_i) - phi_R)
        return dpdt

    def dphiI_dt(self, phi_I, t, a, phi_R, f_i):
        # stress sector mass fraction ODE
        k_t = self.k_t0 * self.g(a) # translational efficiency
        dpdt = k_t * (phi_R - self.phiR_min) * (f_i - phi_I)
        return dpdt

    def dpFidt(self, t, f_i):
        drift = self.f_i
        dpdt = -(1/self.tau_u)*(f_i - drift)
        return dpdt

    def dAAdt(self, a, t, phi_R, phiR_max, phi_I):
        # amino acid concentration ODE (variable nutrient conc.(c))
        k_n = self.k_n0 * self.f(a) # nutritional efficiency
        k_t = self.k_t0 * self.g(a) # translational efficiency

        dadt = k_n * (phiR_max - phi_R - phi_I) - k_t * (phi_R - self.phiR_min)
        return dadt

    def dXdt(self, X, t, a, phi_R, V, phiR_max, f_i):
        # division protein ODE
        dxdt = self.f_X(a,phiR_max,f_i) * self.GrowthRate(a, phi_R) * V - self.mu * X
        return dxdt

    def dVdt(self, V, t, a, phi_R):
        # cell volume ODE
        dvdt = self.GrowthRate(a, phi_R) * V
        return dvdt

    def dpRmdt(self, t, phiR_max):
        drift = self.phiRmax_mean
        dpdt = -(1/self.tau)*(phiR_max - drift)
        return dpdt

    def MultiIntegrate(self, Species, t, dt):
        # numerically solve via Euler-Maruyama method
        phiR_i,a_i,X_i,V_i,phiRmax_i,phiI_i,fI_i = Species

        phi_R = phiR_i + self.dphiR_dt(phiR_i, t, a_i, phiRmax_i,fI_i)*dt
        phi_I = phiI_i + self.dphiI_dt(phiI_i, t, a_i, phiR_i, fI_i)*dt
        a = a_i + self.dAAdt(a_i, t, phiR_i, phiRmax_i, phiI_i)*dt
        # ensure that amino acid conc. is not negative
        a[a < 1e-7] = 1e-7
        #   print('Warning: amino acid conc. went negative and was reset, consider decreasing integration step size')

        X = X_i + self.dXdt(X_i, t, a_i, phiR_i, V_i, phiRmax_i, fI_i)*dt
        V = V_i + self.dVdt(V_i, t, a_i, phiR_i)*dt

        noise = np.sqrt(2*self.sigma_phiRmax) * np.sqrt(dt) * np.random.normal(size=phiRmax_i.shape)
        phiR_max = phiRmax_i + self.dpRmdt(t, phiRmax_i) + noise * phiRmax_i

        noise = np.sqrt(2*self.sigma_u) * np.sqrt(dt) * np.random.normal(size=fI_i.shape)
        f_I = fI_i + self.dpFidt(t, fI_i) + noise * fI_i
        f_I = f_I.clip(0,phiR_max)

        # check to make sure value is physical
        phi_R = np.minimum(phi_R, phiR_max)

        return np.array([phi_R,a,X,V,phiR_max,phi_I,f_I])

    # initialize simulation

    def phiR_ss(self, a):
        # function for phi_R at steady state
        k_n = self.k_n0 * self.f(a) # nutritional efficiency, depends on concentration of nutrients outside cell
        k_t = self.k_t0 * self.g(a) # translational efficiency
        return (k_n*(self.phiRmax_mean-self.f_i) + k_t*self.phiR_min) / (k_n + k_t)

    def func_0(self, x):
        # function for calculating steady state conditions for given parameters
        return [self.phiR_ss(x[0]) - x[1], # x[0]=a, x[1]=phi_R
                self.f_R(x[0],self.phiRmax_mean,self.f_i) - x[1]]

    def initialize(self, rand_param=False):

        if rand_param:
            k_t0 = np.random.normal(loc=self.kt0_mean, scale=0.1*self.kt0_mean)
            self.k_t0 = np.clip(k_t0, 1.5,4)
        else:
            self.k_t0 = self.kt0_mean

        # solving for initial conditions to produce steady state
        a0,phi_R0 = optimize.fsolve(self.func_0, [1e-4, 0.3])
        ls = [a0,phi_R0]
        if not all(val >= 0 for val in ls):
            print(ls)
            raise ValueError(f'Initial values are unphysical, change guess of initial conditions')

        # initializing each array to save initial conditions for each new trajectory
        phiR_birth = np.ones((self.num_cells_init))*phi_R0
        a_birth = np.ones((self.num_cells_init))*a0
        phiRmax_birth = np.ones((self.num_cells_init))*self.phiRmax_mean
        phiI_birth = np.ones((self.num_cells_init))*self.f_i
        fI_birth = np.ones((self.num_cells_init))*self.f_i

        # assigning random initial cell volume, in um^3
        cycle_t = np.log(2) / self.GrowthRate(a0, phi_R0)
        start_t = np.random.uniform(0,cycle_t, self.num_cells_init) # assigning random initial cell volume, in um^3
        birth_size = 1 / self.f_X(a0, self.phiRmax_mean, self.f_i) # average cell size at birth at initial steady-state growth
        V_birth = birth_size * np.exp(self.GrowthRate(a0, phi_R0) * start_t)
        X_birth = self.f_X(a0,self.phiRmax_mean,self.f_i)*V_birth *0.5

        self.init_conditions = np.array([phiR_birth, a_birth, X_birth, V_birth, phiRmax_birth, phiI_birth, fI_birth])
        self.t_start = 0
        self.t_stop = self.delta_t

        if self.num_cells_init == 1:
            self.single_cell_phys_state = self.init_conditions

    def upsample(self, val_array, num_cells_add, clip_high=0.99, clip_low=0):
        samples = np.random.normal(np.mean(val_array), np.std(val_array), num_cells_add)
        val_array = np.concatenate((val_array, samples), -1)
        return val_array.clip(clip_low, clip_high)

    # simulatation implementation

    def simulate_population(self, true_num_cells, n_steps=1600, threshold=3000):

        # unpacking initial conditions for each cell trajectory
        phiR_birth, a_birth, X_birth, V_birth, phiRmax_birth, phiI_birth, fI_birth = self.init_conditions.copy()
        if any(np.isnan(a_birth)) or any(a_birth < 0): # checking to make sure nan values are not present
            print('a_start:',a_birth)
            raise ValueError(f'Simulation error, nan or negative values present')
        num_cells_saved = len(V_birth)

        if num_cells_saved > threshold:
            # downsampling if population exceeds threshold
            row_ids = random.sample(range(num_cells_saved), threshold)
            phiR_birth = phiR_birth[row_ids]
            a_birth = a_birth[row_ids]
            X_birth = X_birth[row_ids]
            V_birth = V_birth[row_ids]
            phiRmax_birth = phiRmax_birth[row_ids]
            phiI_birth = phiI_birth[row_ids]
            fI_birth = fI_birth[row_ids]
            num_cells = threshold
        elif (num_cells_saved < threshold) & (true_num_cells > num_cells_saved):
            # upsampling if population decreased, but is still above number currently being simulated
            num_cells_add = int(np.min((threshold-num_cells_saved, true_num_cells-num_cells_saved)))
            num_cells = int(num_cells_saved+num_cells_add)
            phiR_birth = self.upsample(phiR_birth, num_cells_add, self.phiR_max, self.phiR_min)
            a_birth = self.upsample(a_birth, num_cells_add, clip_low=1e-7)
            X_birth = self.upsample(X_birth, num_cells_add)
            V_birth = self.upsample(V_birth, num_cells_add, np.max(V_birth), np.min(V_birth))
            phiRmax_birth = self.upsample(phiRmax_birth, num_cells_add, np.max(phiRmax_birth), np.min(phiRmax_birth))
            phiI_birth = self.upsample(phiI_birth, num_cells_add, np.max(phiI_birth), np.min(phiI_birth))
            fI_birth = self.upsample(fI_birth, num_cells_add, np.max(fI_birth), np.min(fI_birth))
        else:
            num_cells = num_cells_saved

        iterations = int((self.t_stop - self.t_start)*n_steps)
        t = np.linspace(self.t_start,self.t_stop,iterations)
        dt = (self.t_stop - self.t_start)/iterations
        cell_count = [num_cells]

        species_stack = np.array([phiR_birth, a_birth, X_birth, V_birth, phiRmax_birth, phiI_birth, fI_birth])
        for i in range(1, iterations):

            species_stack = self.MultiIntegrate(species_stack, t[i], dt) # integrating one timestep

            # if cell has accumulated sufficient amount of division proteins, it will divide
            birth_check = species_stack[2,:] >= 1
            if birth_check.sum() != 0:
                r = np.random.normal(0.5, 0.04, birth_check.sum())

                # X_stack_children = species_stack[:,birth_check].copy()
                # X_stack_children[4,:] = 0
                # X_stack_children[5,:] = X_stack_children[5,:] * (1 - r)

                species_stack[2,birth_check] = 0
                species_stack[3,birth_check] = species_stack[3,birth_check] * r
                species_stack[4,birth_check] = species_stack[4,birth_check] * np.random.normal(1, self.ribo_div_noise, birth_check.sum())
            species_stack[4,:] = species_stack[4,:].clip(0.3,0.8)

                # species_stack = np.concatenate((species_stack, X_stack_children), -1)

            cell_count.append(species_stack.shape[1])


            if self.num_cells_init == 1:
                self.single_cell_phys_state = np.concatenate((self.single_cell_phys_state, species_stack), axis=-1)

        self.init_conditions = species_stack.copy()
        self.t_start = self.t_stop
        self.t_stop += self.delta_t

        if true_num_cells > threshold:
            true_num_cells_next = np.round(true_num_cells * (cell_count[-1] / num_cells))
        else:
            true_num_cells_next = cell_count[-1]

        # calculating growth rate and activity
        growth_rate = self.GrowthRate(species_stack[1,:], species_stack[0,:])
        activity = species_stack[6,:] * growth_rate

        return t[-1], true_num_cells_next, growth_rate, activity


if __name__ == '__main__':
    pass