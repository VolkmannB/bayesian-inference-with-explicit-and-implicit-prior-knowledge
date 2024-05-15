import numpy as np
from tqdm import tqdm
import jax

import plotly.graph_objects as go
from plotly.subplots import make_subplots


from src.SingleMassOscillator import F_spring, F_damper


def cm2in(cm):
    return cm*0.3937007874



def generate_Animation(X, Y, F_sd, Sigma_X, Sigma_F, weights, H, W, CW, time, dpi, width, fps, filne_name='test.html'):
    
    # create figure
    width = cm2in(width)
    fig = make_subplots(2, 2)
       
    # create point grid
    delta = 0.25
    x_plt = np.arange(-5., 5., delta)
    dx_plt = np.arange(-5., 5., delta)
    grid_x, grid_y = np.meshgrid(x_plt, dx_plt, indexing='xy')
    points = np.dstack([grid_x, grid_y])
    
    # state trajectory as gaussian
    X_pred = np.sum(Sigma_X * weights[...,None], axis=1)
    F_pred = np.sum(Sigma_F * weights, axis=1)

    # calculate ground truth for spring damper force
    grid_F = F_spring(grid_x) + F_damper(grid_y)
    
    # resample time
    samples = 10
    time = time[0:-1:samples]
    
    # calculate predictions for spring damper force
    phi = jax.vmap(H)(points.reshape((points.shape[0]*points.shape[1],2)))
    F_GP = np.einsum('...n,mn->...m', phi, W[0:-1:samples,...])
    F_GP = F_GP.reshape((points.shape[0], points.shape[1], len(time)))
    
    # error between mean of prediction and ground truth
    e = np.abs(grid_F[...,None] - F_GP)
    
    
    ######################## These plots get updated by frames
    
    # plot error
    fig.add_trace(
        go.Scatter(
            x=grid_x.flatten(),
            y=grid_y.flatten(),
            mode='markers',
            marker={
                'color': e[...,0].flatten(), 
                'colorscale': 'Viridis', 
                'cmin':0, 
                'cmid':2.5, 
                'cmax':10,
                'size':8,
                'colorbar': {
                    'thickness':20, 
                    'title': {
                        'text':'Prediction Error' , 
                        'font': {'size': 24}
                    }
                }
            }
        ),
        row=1,
        col=1
    )
    
    # plot ensamble
    fig.add_trace(
        go.Scatter(
            y=Sigma_X[0,:,1],
            x=Sigma_X[0,:,0],
            mode='markers',
            marker={'color': 'red', 'size':5}
        ),
        row=1,
        col=1
    )
    
    # plot time markers
    fig.add_trace(
        go.Scatter(
            x=[time[0], time[0]], 
            y=[-3, 3],
            mode='lines',
            line=dict(color='red', width=2)
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[time[0], time[0]], 
            y=[-7, 7],
            mode='lines',
            line=dict(color='red', width=2)
        ),
        row=2,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[time[0], time[0]], 
            y=[-50, 60],
            mode='lines',
            line=dict(color='red', width=2)
        ),
        row=1,
        col=2
    )
    
    ########################
    
    # plot position
    fig.add_trace(
        go.Scatter(
            x=time, 
            y=X[0:-1:samples,0],
            mode='lines',
            line=dict(color='blue', width=4)
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=time, 
            y=X_pred[0:-1:samples,0],
            mode='lines',
            line=dict(color='orange', width=4, dash='dot')
        ),
        row=2,
        col=1
    )
    
    # plot velocity
    fig.add_trace(
        go.Scatter(
            x=time, 
            y=X[0:-1:samples,1],
            mode='lines',
            line=dict(color='blue', width=4)
        ),
        row=2,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            x=time, 
            y=X_pred[0:-1:samples,1],
            mode='lines',
            line=dict(color='orange', width=4, dash='dot')
        ),
        row=2,
        col=2
    )
    
    # plot force
    fig.add_trace(
        go.Scatter(
            x=time, 
            y=F_sd[0:-1:samples],
            mode='lines',
            line=dict(color='blue', width=4)
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            x=time, 
            y=F_pred[0:-1:samples],
            mode='lines',
            line=dict(color='orange', width=4, dash='dot')
        ),
        row=1,
        col=2
    )
    
    # generate Frames
    Sigma_X = Sigma_X[0:-1:samples,...]
    frames = [dict(
        name = str(k),
        data = [
            go.Scatter(
                marker={'color': e[...,k].flatten()}),
            go.Scatter(
                x=Sigma_X[k,:,0],
                y=Sigma_X[k,:,1],
                mode='markers'),
            go.Scatter(
                x=[time[k], time[k]], 
                y=[-3, 3],
                mode='lines',
                line=dict(color='red', width=2)),
            go.Scatter(
                x=[time[k], time[k]], 
                y=[-7, 7],
                mode='lines',
                line=dict(color='red', width=2)),
            go.Scatter(
                x=[time[k], time[k]], 
                y=[-50, 60],
                mode='lines',
                line=dict(color='red', width=2))
            ],
        traces=list(range(5))
        ) for k in tqdm(np.arange(1,len(time)), desc='Generating animated Frames')]
    
    # # add frames
    fig.update(frames=frames)
    
    # set axis limits
    fig.update_xaxes(range=[-5, 5], row=1, col=1)
    fig.update_yaxes(range=[-5, 5], row=1, col=1)
    
    fig.update_xaxes(range=[0, time[-1]], row=2, col=1)
    fig.update_xaxes(range=[0, time[-1]], row=2, col=2)
    fig.update_xaxes(range=[0, time[-1]], row=1, col=2)
    
    fig.update_yaxes(range=[-3, 3], row=2, col=1)
    fig.update_yaxes(range=[-7, 7], row=2, col=2)
    fig.update_yaxes(range=[-50, 60], row=1, col=2)
    
    # axis descriptions
    fig.update_xaxes(title_text='Time in (s)', title_font_size=24, row=2, col=1)
    fig.update_xaxes(title_text='Time in (s)', title_font_size=24, row=2, col=2)
    fig.update_xaxes(title_text='Time in (s)', title_font_size=24, row=1, col=2)
    fig.update_xaxes(title_text='Position in (m)', title_font_size=24, row=1, col=1)
    fig.update_yaxes(title_text='Velocity in (m/s)', title_font_size=24, row=1, col=1)
    fig.update_yaxes(title_text='Position in (m)', title_font_size=24, row=2, col=1)
    fig.update_yaxes(title_text='Velocity in (m/s)', title_font_size=24, row=2, col=2)
    fig.update_yaxes(title_text='Force in (N)', title_font_size=24, row=1, col=2)
    
    # axis font soze
    fig.update_xaxes(tickfont_size=24)
    fig.update_yaxes(tickfont_size=24)
    
    # update layout
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
    
    # save file
    fig.write_html(filne_name)

    return fig
