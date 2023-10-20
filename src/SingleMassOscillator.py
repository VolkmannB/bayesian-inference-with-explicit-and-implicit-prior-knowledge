import numpy as np



def F_spring(x, c1, c2):
    return c1*x+c2*x**3



def F_damper(dx, d1, d2):
    return d1*dx * (1/(1+d2*dx*np.tanh(dx)))



def dx(x, F, m, c1, c2, d1, d2):
    
    F_s = F_spring(x[0], c1, c2)
    F_d= F_damper(x[1], d1, d2)
    
    return np.array(
        [x[1], -F_s/m - F_d/m + F/m + 9.81]
    )



def f_x(x, F, m, c1, c2, d1, d2, dt):
    
    # Runge-Kutta 4
    k1 = dx(x, F, m, c1, c2, d1, d2)
    k2 = dx(x+dt/2.0*k1, F, m, c1, c2, d1, d2)
    k3 = dx(x+dt/2.0*k2, F, m, c1, c2, d1, d2)
    k4 = dx(x+dt*k3, F, m, c1, c2, d1, d2)
    x = x + dt/6.0*(k1+2*k2+2*k3+k4) 
    
    return x



def f_model(s, v, F, F_sd, m, dt):
    
    
    dx = np.concatenate(
        [v, -F_sd/m + F/m + 9.81], dim=-1
    )
    
    x = np.concatenate(
        [s, v], dim=-1
    )
    
    return x + dt*dx



class SingleMassOscillator():
    
    def __init__(
        self, 
        x0 = np.array([0., 0.]),
        Q = np.diag([5e-6, 5e-7]),
        R = np.array([[1e-2]]),
        m=2., 
        c1=10., 
        c2=2., 
        d1=0.4, 
        d2=0.4
        ) -> None:
        
        if m <= 0:
            raise ValueError("Mass must be positive")
        if c1 <=0 or c2 < 0:
            raise ValueError("Spring coefficients ust be c1 >= 0 or c2 > 0")
        if d1 <=0 or d2 < 0:
            raise ValueError("Spring coefficients ust be d1 >= 0 or d2 > 0")
        
        self.m = m
        self.c1 = c1
        self.c2 = c2
        self.d1 = d1
        self.d2 = d2
        
        self.x = x0
        
        self.Q = Q
        self.R = R
    
    
    
    def update(self, F: float, dt: float):
        
        m = self.m
        c1 = self.c1
        c2 = self.c2
        d1 = self.d1
        d2 = self.d2
        
        # Runge-Kutta 4
        k1 = dx(self.x, F, m, c1, c2, d1, d2)
        k2 = dx(self.x+dt/2.0*k1, F, m, c1, c2, d1, d2)
        k3 = dx(self.x+dt/2.0*k2, F, m, c1, c2, d1, d2)
        k4 = dx(self.x+dt*k3, F, m, c1, c2, d1, d2)
        self.x = self.x + dt/6.0*(k1+2*k2+2*k3+k4) + np.random.randn(self.x.shape) @ np.linalg.cholesky(self.Q).T
    
    
    
    def measurent(self):
        return self.x[0] + np.random.randn(1)*self.R