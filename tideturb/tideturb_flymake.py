"""This is a model of a turbidity current influenced by tidal flows in
 a submarine canyon. The two-layer shallow water equation system is
 employed. The upper layer is an ambient water, and the lower layer
 is a turbidity current.

.. codeauthor:: Hajime Naruse

Example
--------
from tideturb import TwoLayerTurbidityCurrent, Grid
from matplotlib import pyplot as plt

grid = Grid(number_of_grids=100, spacing=10.0)
grid.eta = grid.x * -0.01
tc = TwoLayerTurbidityCurrent(grid=grid, turb_vel=1.0, ambient_vel=-0.3, turb_thick=5.0, ambient_thick=100.0)
steps = 500
for i in range(steps):
    tc.plot()
    plt.savefig('tidal_negative_{:04d}'.format(i))
    tc.run_one_step(dt=3.0)
    print("", end='\r')
    print('{:.1f}% finished.'.format(i/steps*100), end='\r')
    
plt.show()


"""

import numpy as np
import matplotlib.pyplot as plt
import ipdb

class Grid():
    
    def __init__(self, number_of_grids = 100, start=0.0, end = None, spacing=1.0):
        """Grids used for the TwoLayerTurbidityCurrent
        
           Parameters
           ----------------
           number_of_grids : int, optional
                Number of grid nodes in the computational domain.

           start : float, optional
                A value of x-coordinate at the landward end of the
                computational domain (m).
            
           end : float, optional
                A value of x-coordinate at the landward end of the
                computational domain (m).
           
           spacing : float, optional
                Spacing of the grid nodes. This is ignored when the
                parameter "end" is specified.
        """
        self.number_of_grids = number_of_grids
        self.dx = spacing
        if end is None:
            self.x = np.arange(start,
                               start + spacing * number_of_grids, spacing)
        else:
            self.x = np.arange(start, end, spacing) 
        
        # bed elevation
        self.eta = np.zeros(self.x.shape)
        
        # flow parameters
        self.U_a = np.zeros(self.x.shape) # velocity of ambient water
        self.U_t = np.zeros(self.x.shape) # velocity of a turbidity current
        self.h_a = np.zeros(self.x.shape) # height of ambient water
        self.h_t = np.zeros(self.x.shape) # height of a turbidity current
        self.C = np.zeros(self.x.shape) # sediment concentration
        
        # indeces of core grids (excluding boundary grids)
        self.core_nodes = np.arange(1, len(self.x) - 1, dtype='int')
        self.core_links = np.arange(1, len(self.x) - 2, dtype='int')
        
class TwoLayerTurbidityCurrent():
    """Two layer model of a turbidity current and an overlying tidal current
    """

    def __init__(
                self,
                grid=None,
                ambient_vel = 0.0,
                ambient_thick = 20,
                turb_vel = 1.0,
                turb_thick = 10,
                concentration = 0.01,
                R = 1.65,
                g = 9.81,
                Cf = 0.004,
                nu = 1.010 * 10 ** -3,
                h_init = 0.001,
                C_init = 0.0001,
                alpha = 0.0001,
                ):
        """ Constractor for TwoLayerTurbidityCurrent
        
            Parameters
            -----------
            grid : Grid, optional
               Grid object that is used for calculation. If this parameter
               is not specified, the default values of Grid object are used.
               
            ambient_vel : float, optional
               Flow velocity of ambient water (supposing tidal current) at
               the upstream end (m/s).
               
            ambient_thick : float, optional
               Initial thickness of ambient water (m).
            
            turb_vel : float, optional
               Velocity of a turbidity current at the upstream end (m/s).
               
            turb_thick : float, optional
               Thickness of a turbidity current at the upstream end (m/s).
               
            concentration : float, optional
               Sediment concentration in a turbidity current.
               
            R : float, optional
               Submerged specific density of sediment particles. 1.65 for
               quartz grains.
               
            g : float, optional
               Gravity acceleration
               
            Cf : float, optional
               Bed friction coefficient
               
            nu : float, optional
               Eddy visosity at the interface between two layers
               
            h_init : float, optional
               Dummy flow thickness of turbidity current. This is needed for
               numerical stability.
               
        """
        try:
        
            # set a grid
            if grid is None:
                self.grid = Grid()
            else:
                self.grid = grid
        
            # store parameters
            self.R = R
            self.g = g
            self.Cf = Cf
            self.nu = nu
            self.h_init = h_init
            self.C_init = C_init
            self.ambient_vel = ambient_vel
            self.ambient_thick = ambient_thick
            self.turb_vel = turb_vel
            self.turb_thick = turb_thick
            self.dx = self.grid.dx
            self.alpha = alpha
            self.dt = 0.1
            self.elapsed_time = 0.0
        
        except Exception as exc:
            print(type(exc))    # the exception instance
            print(exc.args)     # arguments stored in .args
            print(exc)

        # Set main variables
        # The subscript "node" denotes variables at grids. The subscript "link" denotes variables at half-staggered point between grids.
        # This model employs staggered grids. Flow heights are calculated at "nodes", and flow velocities are calculated at "link".
        # The "node" values and "link" values are mapped each other by averaging.
        # h_node[0,:] is the ambient flow height h_a, and h_node[1, :] is the height of the turbidity current.
        # U_link[0,:] is the ambient flow velocity U_a, and U_link[1, :] is the velocity of the turbidity current.
        self.h_node = np.zeros([2, self.grid.x.shape[0]]) # h_a and h_t values at nodes
        self.h_link = np.zeros([2, self.grid.x.shape[0] - 1]) # h_a and h_t values at link
        self.U_node = np.zeros([2, self.grid.x.shape[0]]) # U_a and U_t values at nodes
        self.U_link = np.zeros([2, self.grid.x.shape[0] - 1]) # U_a and U_t values at nodes
        self.C_node = np.zeros([2, self.grid.x.shape[0]])  # concentration values at nodes
        self.C_link = np.zeros([2, self.grid.x.shape[0] - 1]) #  concentration values at link
                
        # spatial derivatives
        self.dhdx = np.zeros(self.h_node.shape) # spatial derivatives of heights
        self.dUdx = np.zeros(self.U_link.shape) # spatial derivatives of velocities
        self.dCdx = np.zeros(self.C_node.shape) # spatial derivatives of concentration
        
        # non advection terms
        self.G_h = np.zeros(self.h_node.shape) # non-advection term for h_a and h_t
        self.G_U = np.zeros(self.U_link.shape) # non-advection term for U_a and U_t
        self.G_C = np.zeros(self.C_node.shape) # non-advection term for C
        
        # Set core nodes and links. Only these core grids are used for calculation.
        # Other nodes and links are used to describe boundary conditions.
        core_nodes = np.tile(self.grid.core_nodes, (self.h_node.shape[0], 1))
        core_links = np.tile(self.grid.core_links, (self.U_link.shape[0], 1))
        self.core_nodes = [np.array([np.arange(self.h_node.shape[0], dtype='int')]).T * np.ones(core_nodes.shape, dtype='int'), core_nodes]
        self.core_links = [np.array([np.arange(self.U_link.shape[0], dtype='int')]).T * np.ones(core_links.shape, dtype='int'), core_links]            

        # Set initial and boundary conditions
        self.h_node[1, 0] = turb_thick
        self.h_node[1, 1:] = h_init * np.ones(self.h_node[1, 1:].shape)
        self.h_node[0, :] = ambient_thick + turb_thick - self.grid.eta - self.h_node[1,:] # initial water surface is flat
        self.h_link[0, :] = (self.h_node[0,:-1] + self.h_node[0,1:]) / 2.
        self.U_link[0, :] = ambient_vel * ambient_thick / self.h_link[0, :] 
        self.U_link[1, 0] = turb_vel
        self.C_node[1, 0] = concentration
        self.C_node[1, 1:] = np.ones(self.C_node.shape[1] - 1) * self.C_init # sediment concentration is defined at node
        self.eta_node = self.grid.eta
        self.eta_link = (self.eta_node[0:-1] + self.eta_node[1:]) / 2.
        
        # variables to store calculation results temporary
        self.h_temp = self.h_node.copy()
        self.U_temp = self.U_link.copy()
        self.C_temp = self.C_node.copy()
        self.dhdx_temp = self.dhdx.copy()
        self.dUdx_temp = self.dUdx.copy()
        self.dCdx_temp = self.dCdx.copy()

        # Map node and link values each other
        self.update_values()

    def calc_time_step(self):
        """calculate time step length based on CFL condition with a safe rate alpha
        
        Return
        ---------
        dt : float
            A time step length to be used as dt_local.
            
        """
        dt = self.alpha * self.dx / np.amax(np.array([np.amax(np.abs(self.U_link)), 1.0]))
        return dt
    
    def run_one_step(self, dt=None):
        """ Calculate one step. The model runs for dt seconds.
            Internally, it uses a local time step that is determined by CFL condition.
            
            Parameter
            -----------
            dt : float, optional
               Time that will elapsed by this step (s). If not specified, internal time step obtained by the function calc_time_step() is used.
            
        """
        
        # set the local mesurement of elapsed time and the duration of this run
        elapsed_time_local = 0.0
        if dt is None:
            dt = self.calc_time_step()
        
        # Loop until prescribed time
        while elapsed_time_local < dt:
            # set time step length
            dt_local = self.calc_time_step()
            if elapsed_time_local + dt_local > dt:
                dt_local = dt - elapsed_time_local
            
            # find upcurrent and downcurrent ids
            up_node, down_node = self.find_current_direction(self.U_node, self.core_nodes)
            up_link, down_link = self.find_current_direction(self.U_link, self.core_links)

            # Calculate advection phases of h and U respectively
            self.cip_1d_advection(
                self.h_node,
                self.dhdx,
                self.U_node,
                self.core_nodes,
                up_node,
                down_node,
                self.dx,
                dt_local,
                out_f=self.h_temp,
                out_dfdx=self.dhdx_temp)
            self.cip_1d_advection(
                self.U_link,
                self.dUdx,
                self.U_link,
                self.core_links,
                up_link,
                down_link,
                self.dx,
                dt_local,
                out_f=self.U_temp,
                out_dfdx=self.dUdx_temp)
            self.cip_1d_advection(
                self.C_node,
                self.dCdx,
                self.U_node,
                self.core_nodes,
                up_node,
                down_node,
                self.dx,
                dt_local,
                out_f=self.C_temp,
                out_dfdx=self.dCdx_temp)
            self.update_values()
            
            # Calculate non-advection phase of h and U
            self.calc_G_h(out_G=self.G_h)
            self.calc_G_U(out_G=self.G_U)
            self.calc_G_C(out_G=self.G_C)
            self.cip_1d_nonadvection(self.h_node, self.dhdx, self.U_node, self.G_h, self.core_nodes, up_node, down_node, self.dx, dt_local, out_f= self.h_temp, out_dfdx=self.dhdx_temp)
            self.cip_1d_nonadvection(self.U_link, self.dUdx, self.U_link, self.G_U, self.core_links, up_link, down_link, self.dx, dt_local, out_f=self.U_temp, out_dfdx=self.dUdx_temp)
            self.cip_1d_nonadvection(self.C_node, self.dCdx, self.U_node, self.G_C, self.core_nodes, up_node, down_node, self.dx, dt_local, out_f=self.C_temp, out_dfdx=self.dCdx_temp)
            self.update_values()
            
            # increment the time step
            elapsed_time_local = elapsed_time_local + dt_local
        
        # increment the total elapsed time
        self.elapsed_time = self.elapsed_time + elapsed_time_local
    
    def get_ew(self, U, h, C):
        """ calculate entrainemnt coefficient of ambient water to a turbidity current layer
        
            Parameters
            ----------
            U : ndarray
               Flow velocities of ambient water and a turbidity current. Row 0 is for an ambient water, and Row 1 is for a turbidity current. 
            h : ndarray
               Flow heights of ambient water and a turbidity current. Row 0 is an ambient water, and Row 1 is a turbidity current.
            C : ndarray
               Sediment concentration
               
            Returns
            ---------
            e_w : ndarray
               Entrainment coefficient of ambient water
               
        """
        Ri = np.zeros(U.shape[1])
        e_w = np.zeros(U.shape[1])
    
        U_abs = np.abs(U[0,:] - U[1,:])
        flow_exist = np.where((h[1, :] > self.h_init * 10) & (U_abs > 0))
        
        Ri[flow_exist] = self.R * self.g * C[1, flow_exist] * h[1, flow_exist] / U_abs[flow_exist] ** 2
        e_w[flow_exist] = 0.075 / np.sqrt(1 + 718. + Ri[flow_exist] ** 2.4) # Parker et al. (1987)
        
        return e_w
    
    def find_current_direction(self, u, core):
        """ check flow directions and return upcurrent and downcurrent indeces
        
            Parameters
            -----------
            u : ndarray
                flow velocity of an ambient water and a turbidity current layers. Row 0 is for an ambient water, and Row 1 is for a turbidity current.
            core : ndarray
                Grid indeces showing "core" of the calculation domain. Boundary grids are excluded.
            
            Returns
            -----------
            up : ndarray
                Indeces of nodes or links that locate upcurrent to the core grids
            down : ndarray
                Indeces of nodes or links that locate downcurrent to the core grids
                
        """
        # Firstly, flow velocity is assumed to be positive everywhere
        up = [core[0], core[1] - 1]
        down = [core[0], core[1] + 1]
        
        # find the negative values in the flow velocity arrays
        negative_u_index = np.where(u[core] < 0)
        up[1][negative_u_index] = core[1][negative_u_index] + 1
        down[1][negative_u_index] = core[1][negative_u_index] - 1
        
        return up, down

    def update_values(self):
        """ Update values stored in grids, and process boundary conditions
            Mapping values at nodes and links each other
        """
        # process boundary conditions
        self.h_node[:, -1] = (self.h_node[:, -2] + self.h_node[:, -3] + self.h_node[:, -4]) / 3 
        self.C_node[:, -1] = (self.C_node[:, -2] + self.C_node[:, -3] + self.C_node[:, -4]) / 3
        self.U_link[:, -1] = (self.U_link[:, -2] + self.U_link[:, -3] + self.U_link[:, -4]) / 3
        
        # copy temporal values to main variables
        self.h_node[:,:] = self.h_temp[:,:]
        self.U_link[:,:] = self.U_temp[:,:]
        self.C_node[:, :] = self.C_temp[:, :]
        self.dhdx[:,:] = self.dhdx_temp[:,:]
        self.dUdx[:,:] = self.dUdx_temp[:,:]
        self.dCdx[:] = self.dCdx_temp[:]
        
        # update node and link values
        self.h_node[self.h_node < self.h_init] = self.h_init
        self.C_node[1, self.C_node[1,:] < self.C_init] = self.C_init
        self.h_link[:, :] = (self.h_node[:, 0:-1] + self.h_node[:, 1:]) / 2.
        self.U_node[:, 1:-1] = (self.U_link[:, 0:-1] + self.U_link[:, 1:]) / 2.
        self.U_node[:, 0] = self.U_link[:, 0]
        self.U_node[:, -1] = self.U_link[:, -1]
        self.C_link[:, :] = (self.C_node[:, 0:-1] + self.C_node[:, 1:]) / 2.
        
        # update values in the grid
        self.grid.h_a = self.h_node[0, :]
        self.grid.h_t = self.h_node[1, :]
        self.grid.U_a = self.U_node[0, :]
        self.grid.U_t = self.U_node[1, :]
        self.grid.C = self.C_node[1, :]
    
    def calc_G_h(self, out_G=None):
        """ Calculate non-advection terms for flow heights
            
            Parameter
            -----------
            out_G : ndarray, optional
                An array to return the calculation result
            
            Return
            ------------
            out_G : ndarray
                Calculation result
        """
        if out_G is None:
            out_G = np.zeros(self.G_h.shape)
        
        # set parameters
        h_a = self.h_node[0,:]
        h_t = self.h_node[1,:]
        U_a = self.U_node[0,:]
        U_t = self.U_node[1,:]
        e_w = self.get_ew(self.U_node, self.h_node, self.C_node)
        core = self.core_nodes[1][0,:]
        up = core - 1
        down = core + 1
        dx = self.dx
        
        # calculate non-advection terms
        out_G[0, core] =  - e_w[core] * np.abs(U_t[core] - U_a[core]) - h_a[core] * (U_a[down] - U_a[up]) / (2 * dx) # G_ha
        out_G[1, core] =  e_w[core] * np.abs(U_t[core] - U_a[core]) \
                        - h_t[core] * (U_t[down] - U_t[up]) / (2 * dx) # G_ht        
        
        return out_G
    
    def calc_G_U(self, out_G=None):
        """ Calculate non-advection terms of flow velocities
        
            Parameter
            -----------
            out_G : ndarray, optional
                An array to return the calculation result
            
            Return
            ------------
            out_G : ndarray
                Calculation result
        """
        
        if out_G is None:
            out_G = np.zeros(self.G_U.shape)
        
        # set parameters
        h_a = self.h_link[0,:]
        h_t = self.h_link[1,:]
        U_a = self.U_link[0,:]
        U_t = self.U_link[1,:]
        C = self.C_link[1, :]
        eta = self.eta_link
        e_w = self.get_ew(self.U_link, self.h_link, self.C_link)
        R = self.R
        g = self.g
        Cf = self.Cf
        nu = self.nu
        core = self.core_links[1][0,:]
        up = core - 1
        down = core + 1
        dx = self.dx
             
        # calculate non-advection terms        
        out_G[0, core] = - g * (eta[down] - eta[up]) / (2 * dx) - g * ((h_a[down] + h_t[down]) - (h_a[up] + h_t[up])) / (2 * dx) \
                        - 2 * nu / h_a[core] * (U_a[core] - U_t[core]) / (h_a[core] + h_t[core]) \
                        + (e_w[core] * U_t[core] * U_a[core]) / h_a[core]
        out_G[1, core] = - (1 + R * C[core]) * g * (eta[down] - eta[up]) / (2 * dx) \
                        - g * ((h_a[down] + h_t[down]) - (h_a[up] + h_t[up])) / (2 * dx) \
                        - R * C[core] * g * (h_t[down] - h_t[up]) / (2 * dx) \
                        + 2 * nu / h_t[core] * (U_a[core] - U_t[core]) / (h_a[core] + h_t[core]) \
                        - Cf * U_t[core] * np.abs(U_t[core]) / h_t[core] \
                        - e_w[core] * U_t[core] ** 2 / h_t [core]
        
        return out_G

    def calc_G_C(self, out_G=None):
        """ Calculate non-advection terms for concentration
            
            Parameter
            -----------
            out_G : ndarray, optional
                An array to return the calculation result
            
            Return
            ------------
            out_G : ndarray
                Calculation result
        """
        if out_G is None:
            out_G = np.zeros(self.G_C.shape)
        
        # set parameters
        h_t = self.h_node[1,:]
        U_a = self.U_node[0,:]
        U_t = self.U_node[1,:]
        C = self.C_node[1, :]
        e_w = self.get_ew(self.U_node, self.h_node, self.C_node)
        core = self.core_nodes[1][0,:]

        # settling and entrainment of sediment are not implemented
        w_s = 0
        e_s = 0
        r_0 = 0
        
        # calculate non-advection terms
        out_G[1, core] = (w_s * (e_s - r_0 * C[core]) - e_w[core] * C[core] * np.abs(U_t[core] - U_a[core])) / h_t[core] # G_C
        
        return out_G

    
    
    def cip_1d_advection(self, f, dfdx, u, core, up, down, dx, dt, out_f=None, out_dfdx=None):
        """ calculate 1 step of advection phase by CIP method
        """
        
        if out_f is None:
            out_f = np.zeros(f.shape)
            out_f[:,0] = f[:,0]
            out_f[:,-1]  = f[:,-1]
        if out_dfdx is None:
            out_dfdx = np.zeros(f.shape)
            out_dfdx[:,0] = dfdx[:,0]
            out_dfdx[:,-1]  = dfdx[:,-1]

        # advection phase
        D = -np.where(u > 0., 1.0, -1.0) * dx
        xi = -u * dt
        a = (dfdx[core] + dfdx[up]) / (D[core] ** 2)\
            + 2 * (f[core] - f[up]) / (D[core] ** 3)
        b = 3 * (f[up] - f[core]) / (D[core] ** 2)\
            - (2 * dfdx[core] + dfdx[up]) / D[core]
        out_f[core] = a * (xi[core] ** 3) + b * (xi[core] ** 2)\
            + dfdx[core] * xi[core] + f[core]
        out_dfdx[core] = 3 * a * (xi[core] ** 2) + 2 * b * xi[core]\
            + dfdx[core]
        
        return out_f, out_dfdx

    def cip_1d_nonadvection(self, f, dfdx, u, G, core, up, down, dx, dt, out_f=None, out_dfdx=None):
        """calculate 1 step of non-advection phase by CIP method
        """
        
        if out_f is None:
            out_f = np.zeros(f.shape)
            out_f[:,0] = f[:,0]
            out_f[:,-1]  = f[:,-1]
        if out_dfdx is None:
            out_dfdx = np.zeros(dfdx.shape)
            out_dfdx[:, 0] = dfdx[:,0]
            out_dfdx[:,-1]  = dfdx[:,-1]

        D = -np.where(u > 0., 1.0, -1.0) * dx
        xi = -u * dt
        
        # non-advection term
        out_f[core] = f[core] + G[core] * dt
        out_dfdx[core] = dfdx[core] + ((out_f[down] - f[down]) - (out_f[up] - f[up])) / \
            (-2 * D[core]) - dfdx[core] * \
            (xi[down] - xi[up]) / (2 * D[core])

        return out_f, out_dfdx
        
    def plot(self, xlim=None, ylim=None):
        """ plot results
        """
        
        if xlim is None:
            xlim = [np.min(self.grid.x), np.max(self.grid.x)]
        
        if ylim is None:
            ylim = [np.min(self.grid.eta), np.max(self.grid.h_a)* 1.1]
        
        plt.cla()
        plt.plot(self.grid.x, self.grid.h_t + self.grid.eta, label='turbidity current', color='r')
        plt.plot(self.grid.x, self.grid.h_a + self.grid.h_t + self.grid.eta, label='water surface', color='b')
        plt.plot(self.grid.x, self.grid.eta, label='bed', color='g')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.text(xlim[0], ylim[1] - 20, '$t = $ {:.0f} s.'.format(self.elapsed_time))
        plt.xlabel('Distance (m)')
        plt.ylabel('Elevation (m)')
        plt.legend()
        
        
if __name__ == "__main__":
    grid = Grid(number_of_grids=100, spacing=100.0)
    grid.eta = grid.x * -0.01
    tc = TwoLayerTurbidityCurrent(grid=grid, turb_vel=1.0, ambient_vel=0.3, turb_thick=5.0, ambient_thick=100.0, concentration=0.01, alpha=0.0001)
    steps = 500
    #ipdb.set_trace()
    for i in range(steps):
        tc.plot()
        plt.savefig('test/tidal_flood_{:04d}'.format(i))
        tc.run_one_step(dt=20.0)
        print("", end='\r')
        print('{:.1f}% finished.'.format(i/steps*100), end='\r')
    
    tc.plot()
    plt.savefig('tidal_ebb/tidal_ebb_{:04d}'.format(i))
    plt.show()
