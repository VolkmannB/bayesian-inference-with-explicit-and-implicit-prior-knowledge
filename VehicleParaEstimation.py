import pandas as pd
import pymc as pm
from tqdm import tqdm
import arviz as az
import matplotlib.pyplot as plt
import bokeh.plotting


from src.vehicle.Vehicle import f_x



# load measurments
data = pd.read_csv('src/vehicle/measurment_3.csv', sep=';')
N = 10
    
u = data[['Steering_Angle', 'v_x']].to_numpy()
x = data[['dpsi', 'v_y']].to_numpy()

# Parameters
dt = 0.01
l_f = 1.1625348837209302
l_r = 1.4684651162790696
m = 1720
I_zz = 1827.5431059723351
g = 9.81
mu_x = 0.9

# wrap ssm
state_space_model = lambda x, u, mu, B_f, C_f, E_f, B_r, C_r, E_r: f_x(
            x, u, dt, m, I_zz, l_f, l_r, g, mu_x, 
            mu, B_f, C_f, E_f, B_r, C_r, E_r
            )

# create model
markov_chain = pm.Model()
with markov_chain as mc:
    
    # priors
    # https://x-engineer.org/tire-model-longitudinal-forces/
    # https://uk.mathworks.com/help/physmod/sdl/ref/tireroadinteractionmagicformula.html
    mu = pm.Normal('mu', mu=0.9, sigma=0.1)
    
    C_f = pm.Normal('C_f', mu=1.4, sigma=0.5) # shape
    B_f = pm.Normal('B_f', mu=25, sigma=7) # stiffnes
    E_f = pm.Normal('E_f', mu=0.0, sigma=0.25) # curve
    
    C_r = pm.Normal('C_r', mu=1.4, sigma=0.5) # shape
    B_r = pm.Normal('B_r', mu=25, sigma=7) # stiffnes
    E_r = pm.Normal('E_r', mu=0.0, sigma=0.25) # curve
    
    # initial state
    # x_0 = pm.Normal('x_0', mu=x[0:N,:], sigma=[0.02, 0.15])
    
    states = [x[0]]
    for i in tqdm(range(1, N), desc='Chaining states'):
        state_t = state_space_model(
            states[i-1], u[i], mu, B_f, C_f, E_f, B_r, C_r, E_r
            )
        states.append(state_t)
    
    print('Building likelihood')
    likelihood = pm.Normal('y', mu=states, sigma=[0.02, 0.15], observed=x[1:N+1])
    
    print('Sampling')
    trace = pm.sample(cores=1)


# bokeh.plotting.output_file(filename='Trace_Plot.html', title='Trace Plot')
fig = az.plot_trace(trace, combined=True)
# bokeh.plotting.save(fig)
# bokeh.plotting
plt.show()

summary = az.summary(trace, round_to=2)
print(summary)
summary.to_csv('Posterior_Statistics.csv', sep=';')