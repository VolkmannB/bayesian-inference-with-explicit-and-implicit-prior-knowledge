import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import jax
import jax.numpy as jnp
import functools



from src.RGP import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv
from src.RGP import prior_mniw_updateStatistics, prior_mniw_sampleLikelihood
from src.KalmanFilter import systematic_SISR, squared_error
from src.Plotting import generate_Animation



def algorithm_PF_GibbsSampling_GP(
    N_basis_fcn: int,
    N_xi: int,
    t_end: float,
    dt: float,
    ctrl_input: npt.NDArray,
    Y: npt.NDArray,
    X0: npt.NDArray,
    GP_prior: list[npt.NDArray],
    basis_fcn,
    ssm_fcn,
    measurment_fcn,
    likelihood_fcn,
    noise_fcn):
    
    
    N_particle, N_x = np.shape(X0)
    
    # create time vector
    time = np.arange(0.0,t_end,dt)
    steps = len(time)
    
    # place holders for plotting
    Sigma_X = np.zeros((steps, N_particle, N_x))
    Sigma_Xi = np.zeros((steps, N_particle, N_xi))
    weights = np.ones((steps, N_particle))/N_particle
    GP_mean = np.zeros((steps, N_xi, N_basis_fcn))
    
    # set starting value
    Sigma_X[0,...] = X0
    
    # sufficient statistics of the GP
    GP_model_stats = [
        np.zeros((N_particle, N_basis_fcn, N_xi)),
        np.zeros((N_particle, N_basis_fcn, N_basis_fcn)),
        np.zeros((N_particle, N_xi, N_xi)),
        np.zeros((N_particle,))
    ]
    
    # generate initial key 
    key = jax.random.key(np.random.randint(100, 1000))
    
    
    # simulation loop
    for i in tqdm(range(1,steps), desc="Running Filter"):
        
        ### Step 1: Evaluate basis functions and calculate standard parameters 
        # of the GP
        
        # evaluate basis function
        phi = jax.vmap(
            functools.partial(basis_fcn, ctrl_input=ctrl_input)
            )(
                x=Sigma_X[i-1]
                )
        
        # calculate parameters of GP from prior and sufficient statistics
        GP_para = list(jax.vmap(prior_mniw_2naturalPara_inv)(
            GP_prior[0] + GP_model_stats[0],
            GP_prior[1] + GP_model_stats[1],
            GP_prior[2] + GP_model_stats[2],
            GP_prior[3] + GP_model_stats[3]
        ))
        
        
        
        ### Step 2: According to the algorithm of the auxiliary PF, resample 
        # particles according to the first stage weights
        
        # create auxiliary variable
        xi_aux = jax.vmap(jnp.matmul)(GP_para[0], phi)
        x_aux = jax.vmap(
            functools.partial(ssm_fcn, ctrl_input=ctrl_input[i-1])
            )(
                x=Sigma_X[i-1,...], 
                xi=xi_aux
                )
        
        # calculate first stage weights
        y_aux = jax.vmap(
            functools.partial(measurment_fcn, ctrl_input=ctrl_input[i])
            )(x_aux)
        l = jax.vmap(functools.partial(likelihood_fcn, y=Y[i]))(x=y_aux)
        p = weights[i-1] * l
        p = p/np.sum(p)
        
        #abort
        if np.any(np.isnan(p)):
            print("Particle degeneration at auxiliary weights")
            break
        
        # draw new indices
        u = np.random.rand()
        idx = systematic_SISR(u, p)
        
        # copy statistics
        GP_model_stats[0] = GP_model_stats[0][idx,...]
        GP_model_stats[1] = GP_model_stats[1][idx,...]
        GP_model_stats[2] = GP_model_stats[2][idx,...]
        GP_model_stats[3] = GP_model_stats[3][idx,...]
        GP_para[0] = GP_para[0][idx,...]
        GP_para[1] = GP_para[1][idx,...]
        GP_para[2] = GP_para[2][idx,...]
        GP_para[3] = GP_para[3][idx,...]
        phi = phi[idx]
        
        
        
        ### Step 3: Make a proposal by generating samples from the hirachical 
        # model
        
        # sample from proposal for F
        key, *keys = jax.random.split(key, N_particle+1)
        Sigma_Xi[i-1] = jax.vmap(prior_mniw_sampleLikelihood)(
            key=jnp.asarray(keys),
            M=GP_para[0],
            V=GP_para[1],
            Psi=GP_para[2],
            nu=GP_para[3],
            phi=phi
        )
        
        # sample from proposal for x
        w_x = noise_fcn((N_particle,))
        Sigma_X[i] = jax.vmap(
            functools.partial(ssm_fcn, ctrl_input=ctrl_input[i-1])
            )(
                x=Sigma_X[i-1,idx,:], 
                xi=Sigma_Xi[i-1]
                ) + w_x
            
        
        
        ### Step 4: Update the GP
        
        # apply forgetting operator to statistics for t+1
        GP_model_stats[0] *= 0.999
        GP_model_stats[1] *= 0.999
        GP_model_stats[2] *= 0.999
        GP_model_stats[3] *= 0.999
        
        # update GP parameters
        GP_model_stats = list(jax.vmap(prior_mniw_updateStatistics)(
            *GP_model_stats,
            Sigma_Xi[i-1],
            phi
        ))
        
        
        
        ### Step 5: Calculate new weights
        
        # calculate new weights (measurment update)
        sigma_y = jax.vmap(
            functools.partial(measurment_fcn, ctrl_input=ctrl_input[i])
            )(Sigma_X[i])
        q = jax.vmap(functools.partial(likelihood_fcn, y=Y[i]))(x=sigma_y)
        weights[i] = q / p[idx]
        weights[i] = weights[i]/np.sum(weights[i])
        
        
        
        ### End of Algorithm
        
        # logging
        GP_mean[i,...] = np.einsum('n,n...->...', weights[i], GP_para[0])
        
        #abort
        if np.any(np.isnan(weights[i])):
            print("Particle degeneration at new weights")
            break
    
    
    return Sigma_X, Sigma_Xi, GP_mean, weights, time