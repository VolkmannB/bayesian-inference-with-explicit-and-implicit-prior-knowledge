import numpy as np
from tqdm import tqdm
import jax
import functools

import plotly.graph_objects as go
from plotly.subplots import make_subplots



from src.Vehicle import H_vehicle, vehicle_RBF_ip, mu_y, f_alpha



def generate_Vehicle_Animation(X, Y, u, weights, Sigma_X, Sigma_mu_f, Sigma_mu_r, Sigma_Y, W_f, CW_f, W_r, CW_r, time, model_para, dpi, width, fps, filne_name='vehicle_test.html'):
    
    # create figure
    width = 0.3937007874*width
    fig = make_subplots(4, 2, subplot_titles=('MTF Front', 'MTF Rear', 'Steering Angle', 'Yaw Rate', 'Longitudinal Velocity', 'Lateral Velocity'))
       
    # create point grid
    alpha = np.arange(-20/180*np.pi, 20/180*np.pi, 40/180*np.pi/1000)
    mu_f = mu_y(alpha, model_para['mu'], model_para['B_f'], model_para['C_f'], model_para['E_f'])
    mu_r = mu_y(alpha, model_para['mu'], model_para['B_r'], model_para['C_r'], model_para['E_r'])
    
    # resample time
    samples = 10
    time = time[0:-1:samples]
    
    X = X[0:-1:samples]
    Y = Y[0:-1:samples]
    u = u[0:-1:samples,...]
    W_f = W_f[0:-1:samples,...]
    CW_f = CW_f[0:-1:samples,...]
    W_r = W_r[0:-1:samples,...]
    CW_r = CW_r[0:-1:samples,...]
    Sigma_X = Sigma_X[0:-1:samples,...]
    Sigma_Y = Sigma_Y[0:-1:samples,...]
    weights = weights[0:-1:samples,...]
    Sigma_mu_f = Sigma_mu_f[0:-1:samples,...]
    Sigma_mu_r = Sigma_mu_r[0:-1:samples,...]
    
    # state trajectory as gaussian
    X_pred = np.einsum('ti,tik->tk', weights, Sigma_X)
    Y_pred = np.einsum('ti,tik->tk', weights, Sigma_Y)
    P_pred = np.var(Sigma_X, axis=1)
    mu_f_est = np.einsum('ti,ti->t', weights, Sigma_mu_f)
    mu_r_est = np.einsum('ti,ti->t', weights, Sigma_mu_r)
    
    # limits for axes
    scale = 1.2
    delta_min, delta_max = np.min(u[:,0])*scale, np.max(u[:,0])*scale
    vx_min, vx_max = np.min(u[:,1])*scale, np.max(u[:,1])*scale
    dpsi_min, dpsi_max = np.min(X[:,0])*scale, np.max(X[:,0])*scale
    vy_min, vy_max = np.min(X[:,1])*scale, np.max(X[:,1])*scale
    Y0_min, Y0_max = np.min(X[:,1])*scale, np.max(X[:,1])*scale
    Y1_min, Y1_max = np.min(X[:,1])*scale, np.max(X[:,1])*scale
    
    
    #### Plots
    
    # steering angle
    fig.add_trace(
        go.Scatter(
                x=time,
                y=u[:,0],
                mode='lines',
                name='steering angle',
                line=dict(color='black', width=4)
            ),
            row=2,
            col=1
        )
    
    # longitudinal velocity
    fig.add_trace(
        go.Scatter(
                x=time,
                y=u[:,1],
                mode='lines',
                name='v_x',
                line=dict(color='black', width=4)
            ),
            row=3,
            col=1
        )
    
    # true MTF fornt
    fig.add_trace(
        go.Scatter(
                x=alpha,
                y=mu_f,
                mode='lines',
                name='true mu_yf',
                line=dict(color='blue', width=4)
            ),
            row=1,
            col=1
        )
    
    # true MTF rear
    fig.add_trace(
        go.Scatter(
                x=alpha,
                y=mu_r,
                mode='lines',
                name='true mu_yr',
                line=dict(color='blue', width=4)
            ),
            row=1,
            col=2
        )
    
    # true dpsi
    fig.add_trace(
        go.Scatter(
                x=time,
                y=X[:,0],
                mode='lines',
                name='true dpsi',
                line=dict(color='blue', width=4)
            ),
            row=2,
            col=2
        )
    
    # estimated dpsi
    fig.add_trace(
        go.Scatter(
                x=time,
                y=X_pred[:,0],
                mode='lines',
                name='KF dpsi',
                line=dict(color='orange', width=4, dash='dot')
            ),
            row=2,
            col=2
        )
    
    # true v_y
    fig.add_trace(
        go.Scatter(
                x=time,
                y=X[:,1],
                mode='lines',
                name='true v_y',
                line=dict(color='blue', width=4)
            ),
            row=3,
            col=2
        )
    
    # estimated v_y
    fig.add_trace(
        go.Scatter(
                x=time,
                y=X_pred[:,1],
                mode='lines',
                name='KF v_y',
                line=dict(color='orange', width=4, dash='dot')
            ),
            row=3,
            col=2
        )
    
    # time markers
    fig.add_trace(
        go.Scatter(
            x=[time[0], time[0]], 
            y=[delta_min, delta_max],
            mode='lines',
            line=dict(color='red', width=2)
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[time[0], time[0]], 
            y=[vx_min, vx_max],
            mode='lines',
            line=dict(color='red', width=2)
        ),
        row=3,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[time[0], time[0]], 
            y=[dpsi_min, dpsi_max],
            mode='lines',
            line=dict(color='red', width=2)
        ),
        row=2,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[time[0], time[0]], 
            y=[vy_min, vy_max],
            mode='lines',
            line=dict(color='red', width=2)
        ),
        row=3,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[time[0], time[0]], 
            y=[Y0_min, Y0_max],
            mode='lines',
            line=dict(color='red', width=2)
        ),
        row=4,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[time[0], time[0]], 
            y=[Y1_min, Y1_max],
            mode='lines',
            line=dict(color='red', width=2)
        ),
        row=4,
        col=2
    )
    
    # estimated MTF front
    mu_frac = 10
    mu_f_ = jax.vmap(H_vehicle)(np.atleast_2d(alpha[0:-1:mu_frac]).T) @ W_f[0,...]
    fig.add_trace(
        go.Scatter(
                x=alpha[0:-1:mu_frac],
                y=mu_f_,
                mode='markers',
                name='GP mu_yf',
                marker=dict(color='orange', size=4)
            ),
            row=1,
            col=1
        )
    
    # estimated MTF rear
    mu_r_ = jax.vmap(H_vehicle)(np.atleast_2d(alpha[0:-1:mu_frac]).T) @ W_r[0,...]
    fig.add_trace(
        go.Scatter(
                x=alpha[0:-1:mu_frac],
                y=mu_r_,
                mode='markers',
                name='GP mu_yr',
                marker=dict(color='orange', size=4)
            ),
            row=1,
            col=2
        )
    
    # State Space to MTF scatter front & rear
    alpha_f, alpha_r = jax.vmap(functools.partial(f_alpha, u=u[0], **model_para))(x=Sigma_X[0,:,0:2])
    fig.add_trace(
        go.Scatter(
                x=alpha_f,
                y=Sigma_mu_f[0,:],
                mode='markers',
                name='KF particles',
                marker=dict(color='red', size=5)
            ),
            row=1,
            col=1
        )
    fig.add_trace(
        go.Scatter(
                x=alpha_r,
                y=Sigma_mu_r[0,:],
                mode='markers',
                name='KF particles',
                marker=dict(color='red', size=5)
            ),
            row=1,
            col=2
        )
    
    # Measurment Yaw Rate
    fig.add_trace(
        go.Scatter(
                x=time,
                y=Y[:,0],
                mode='lines',
                name='dpsi meas',
                line=dict(color='blue', width=4)
            ),
            row=4,
            col=1
        )
    
    # predicted Yaw Rate
    fig.add_trace(
        go.Scatter(
                x=time,
                y=Y_pred[:,0],
                mode='lines',
                name='dpsi pred',
                line=dict(color='orange', width=4, dash='dot')
            ),
            row=4,
            col=1
        )
    
    # Measurment a_y
    fig.add_trace(
        go.Scatter(
                x=time,
                y=Y[:,1],
                mode='lines',
                name='a_y meas',
                line=dict(color='blue', width=4)
            ),
            row=4,
            col=2
        )
    
    # predicted a_y
    fig.add_trace(
        go.Scatter(
                x=time,
                y=Y_pred[:,1],
                mode='lines',
                name='a_y pred',
                line=dict(color='orange', width=4, dash='dot')
            ),
            row=4,
            col=2
        )
    
    # generate frames
    frames = []
    for i in tqdm(np.arange(0,len(time)), desc='Generating frames'):
        
        # animated time markers
        ts_delta = go.Scatter(x=[time[i], time[i]], y=[delta_min, delta_max], mode='lines', line=dict(color='red', width=2))
        ts_vx = go.Scatter(x=[time[i], time[i]], y=[vx_min, vx_max], mode='lines', line=dict(color='red', width=2))
        ts_dpsi = go.Scatter(x=[time[i], time[i]], y=[dpsi_min, dpsi_max], mode='lines', line=dict(color='red', width=2))
        ts_vy = go.Scatter(x=[time[i], time[i]], y=[vy_min, vy_max], mode='lines', line=dict(color='red', width=2))
        ts_Y0 = go.Scatter(x=[time[i], time[i]], y=[Y0_min, Y0_max], mode='lines', line=dict(color='red', width=2))
        ts_Y1 = go.Scatter(x=[time[i], time[i]], y=[Y1_min, Y1_max], mode='lines', line=dict(color='red', width=2))
        
        # MTF front
        mu_f_ = jax.vmap(H_vehicle)(np.atleast_2d(alpha[0:-1:mu_frac]).T) @ W_f[i,...]
        MTF_f = go.Scatter(
            x=alpha[0:-1:mu_frac], 
            y=mu_f_
            # mode='lines',
            # line=dict(color='orange', width=4, dash='dot')
            )
        
        # MTF rear
        mu_r_ = jax.vmap(H_vehicle)(np.atleast_2d(alpha[0:-1:mu_frac]).T) @ W_r[i,...]
        MTF_r = go.Scatter(
            x=alpha[0:-1:mu_frac], 
            y=mu_r_
            # mode='lines',
            # line=dict(color='orange', width=4, dash='dot')
            )
        
        # side slip angle
        if i==0:
            i_ = i
        else:
            i_ = i-1
        alpha_f, alpha_r = jax.vmap(functools.partial(f_alpha, u=u[i_], **model_para))(x=Sigma_X[i,:,0:2])
        En_front = go.Scatter(
                    x=alpha_f,
                    y=Sigma_mu_f[i,:],
                    mode='markers',
                    marker=dict(color='red', size=5)
                )
        En_rear = go.Scatter(
                    x=alpha_r,
                    y=Sigma_mu_r[i,:],
                    mode='markers',
                    marker=dict(color='red', size=5)
                )
        
        frame = dict(
            name=str(i), 
            data=[ts_delta, ts_vx, ts_dpsi, ts_vy, ts_Y0, ts_Y1, MTF_f, MTF_r, En_front, En_rear], 
            traces=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            )
        
        frames.append(frame)
    
    # # add frames
    fig.update(frames=frames)
    
    
    
    # Layout
    fig.update_yaxes(range=[delta_min, delta_max], row=2, col=1)
    fig.update_yaxes(range=[vx_min, vx_max], row=3, col=1)
    fig.update_yaxes(range=[dpsi_min, dpsi_max], row=2, col=2)
    fig.update_yaxes(range=[vy_min, vy_max], row=3, col=2)
    fig.update_yaxes(range=[-1.2, 1.2], row=1, col=1)
    fig.update_yaxes(range=[-1.2, 1.2], row=1, col=2)
    fig.update_layout(font_size=24)
    
    updatemenus = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": (time[1]-time[0])*1000, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": (time[1]-time[0])*1000,
                                                                        "easing": "linear"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]
    
    sliders = [
        {
            'yanchor': 'top',
            'xanchor': 'left', 
            'currentvalue': {'font': {'size': 24}, 'prefix': 'Time: ', 'visible': True, 'xanchor': 'right'},
            'transition': {'duration': 300.0, 'easing': 'linear'},
            'pad': {'b': 10, 't': 50}, 
            'len': 0.9, 'x': 0.1, 'y': 0, 
            'steps': [
                {
                    'args': [
                        [k], 
                        {'frame': {'duration': 10, 'easing': 'linear', 'redraw': False},
                        'mode': 'immidiate',
                        'transition': {'duration': 0}}
                        ], 
                        'label': round(time[k],1), 
                        'method': 'animate'
                } for k in range(len(time))]
        }
    ]
    
    fig.update_layout(
        updatemenus=updatemenus,
        sliders=sliders
        )
    fig.update_layout(showlegend=False)
    fig.update_layout(plot_bgcolor='aliceblue')
    fig.update_xaxes(showgrid=True, gridcolor='black', linecolor='black', zerolinecolor='black')
    fig.update_yaxes(showgrid=True, gridcolor='black', linecolor='black', zerolinecolor='black')
    
    # axis descriptions
    fig.update_xaxes(title_text='alpha in (rad)', title_font_size=24, row=1, col=1)
    fig.update_xaxes(title_text='alpha in (rad)', title_font_size=24, row=1, col=2)
    fig.update_xaxes(title_text='Time in (s)', title_font_size=24, row=2, col=1)
    fig.update_xaxes(title_text='Time in (s)', title_font_size=24, row=2, col=2)
    fig.update_xaxes(title_text='Time in (s)', title_font_size=24, row=3, col=1)
    fig.update_xaxes(title_text='Time in (s)', title_font_size=24, row=3, col=2)
    
    fig.update_yaxes(title_text='mu_front', title_font_size=24, row=1, col=1)
    fig.update_yaxes(title_text='mu_rear', title_font_size=24, row=1, col=2)
    fig.update_yaxes(title_text='delta in (rad)', title_font_size=24, row=2, col=1)
    fig.update_yaxes(title_text='dpsi in (rad/s)', title_font_size=24, row=2, col=2)
    fig.update_yaxes(title_text='v_x in (m/s)', title_font_size=24, row=3, col=1)
    fig.update_yaxes(title_text='v_y in (m/s)', title_font_size=24, row=3, col=2)
    
    # save file
    fig.write_html(filne_name)
    
    return fig