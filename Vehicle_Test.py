import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from src.vehicle.Vehicle import f_x, para



t_end = 100
time = np.arange(0, t_end, para[0])

x = np.zeros((time.size, 2))
x[0,:] = np.array([0.0, 0.0])
u = np.zeros((time.size, 2))
u[:,1] = 8.0
u[4000:,0] = 20/180*np.pi
u[6000:,0] = -20/180*np.pi
u[8000:,0] = 0.0


for t in tqdm(np.arange(1, len(time)-1)):
    
    x[t] = f_x(x[t-1], u[t-1], *para)
    


fig = make_subplots(2, 2)

fig.add_trace(
    go.Scatter(
            x=time,
            y=u[:,0],
            mode='lines',
            name='delta'
        ),
        row=1,
        col=1
    )

fig.add_trace(
    go.Scatter(
            x=time,
            y=u[:,1],
            mode='lines',
            name='v_x'
        ),
        row=2,
        col=1
    )

fig.add_trace(
    go.Scatter(
            x=time,
            y=x[:,0],
            mode='lines',
            name='dpsi'
        ),
        row=1,
        col=2
    )

fig.add_trace(
    go.Scatter(
            x=time,
            y=x[:,1],
            mode='lines',
            name='v_y'
        ),
        row=2,
        col=2
    )
    
# axis font soze
fig.update_layout(font_size=24)

fig.show()