"""This is a model of a turbidity current influenced by tidal flows in
 a submarine canyon. The two-layer shallow water equation system is
 employed. The upper layer is an ambient water, and the lower layer
 is a turbidity current.

.. codeauthor:: Hajime Naruse

Example
--------
from tideturb import TwoLayerTurbidityCurrent, Grid
from matplotlib import pyplot as plt

grid = Grid(number_of_grids=100, spacing=100.0)
grid.eta = grid.x * -0.01
tc = TwoLayerTurbidityCurrent(
    grid=grid,
    turb_vel=1.0,
    ambient_vel=0.3,
    turb_thick=5.0,
    ambient_thick=100.0,
    concentration=0.01,
    alpha=0.0001)
steps = 500
for i in range(steps):
    tc.plot()
    plt.savefig('test/tidal_flood_{:04d}'.format(i))
    tc.run_one_step(dt=20.0)
    print("", end='\r')
    print('{:.1f}% finished.'.format(i / steps * 100), end='\r')

tc.plot()
plt.savefig('test/tidal_flood_{:04d}'.format(i))
plt.show()

"""

import numpy as np
import matplotlib.pyplot as plt

import ipdb


class Grid():
    # Grid used for the TwoLayerTurbidityCurrent
    # This class store x coordinates and other flow parameters to output

    def __init__(self, number_of_grids=100, start=0.0, end=None, spacing=1.0):
        """ Constractor of Grid

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

        try:
            # set x coordinate
            self.number_of_grids = number_of_grids
            self.dx = spacing
            if end is None:
                self.x = np.arange(start, start + spacing * number_of_grids,
                                   spacing)
            else:
                self.x = np.arange(start, end, spacing)
        except ValueError as ve:
            print(ve)

        # bed elevation
        self.eta = np.zeros(self.x.shape)

        # flow parameters
        self.U_a = np.zeros(self.x.shape)  # velocity of ambient water
        self.U_t = np.zeros(self.x.shape)  # velocity of a turbidity current
        self.h_a = np.zeros(self.x.shape)  # height of ambient water
        self.h_t = np.zeros(self.x.shape)  # height of a turbidity current
        self.C = np.zeros(self.x.shape)  # sediment concentration

        # indeces of core grids (excluding boundary grids)
        self.core_nodes = np.arange(1, len(self.x) - 1, dtype='int')
        self.core_links = np.arange(1, len(self.x) - 2, dtype='int')


class TwoLayerTurbidityCurrent():
    """Two layer model of a turbidity current and an overlying tidal current
    """

    def __init__(
            self,
            grid=None,
            ambient_vel=0.0,
            ambient_thick=20,
            turb_vel=1.0,
            turb_thick=10,
            concentration=0.01,
            R=1.65,
            g=9.81,
            Cf=0.004,
            nu=1.010 * 10**-3,
            h_init=0.001,
            C_init=0.0001,
            alpha=0.01,
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
            print(type(exc))  # the exception instance
            print(exc.args)  # arguments stored in .args
            print(exc)

        # Set main variables
        # The subscript "node" denotes variables at grids. The subscript
        # "link" denotes variables at half-staggered point between grids.
        # This model employs staggered grids. Flow heights are calculated
        # at "nodes", and flow velocities are calculated at "link".
        # The "node" values and "link" values are mapped each other by
        # averaging.
        # h_node[0,:] is the ambient flow height h_a, and h_node[1, :] is
        # the height of the turbidity current.
        # U_link[0,:] is the ambient flow velocity U_a, and U_link[1, :] is
        # the velocity of the turbidity current.

        # main variables at nodes and links
        self.h_node = np.zeros(self.grid.x.shape[0])
        self.h_link = np.zeros(self.grid.x.shape[0] - 1)
        self.U_node = np.zeros(self.grid.x.shape[0])
        self.U_link = np.zeros(self.grid.x.shape[0] - 1)
        self.C_node = np.zeros(self.grid.x.shape[0])
        self.C_link = np.zeros(self.grid.x.shape[0] - 1)

        # spatial derivatives
        self.dhdx = np.zeros(self.h_node.shape)
        self.dUdx = np.zeros(self.U_link.shape)
        self.dCdx = np.zeros(self.C_node.shape)

        # non advection terms
        self.G_h = np.zeros(self.h_node.shape)
        self.G_U = np.zeros(self.U_link.shape)
        self.G_C = np.zeros(self.C_node.shape)

        # Set core nodes and links. Only these core grids are used for
        # calculation.
        # Other nodes and links are used to describe boundary conditions.
        self.core_nodes = self.grid.core_nodes
        self.core_links = self.grid.core_links

        # Set initial and boundary conditions
        self.h_node[0] = turb_thick
        self.h_node[1:] = h_init * np.ones(self.h_node[1:].shape)
        self.U_link[0] = turb_vel
        self.C_node[0] = concentration

        # sediment concentration is defined at node
        self.C_node[1:] = np.ones(self.C_node.shape[0] - 1) * self.C_init
        self.eta_node = self.grid.eta
        self.eta_link = (self.eta_node[0:-1] + self.eta_node[1:]) / 2.

        # total flow discharge is assumed to be constant
        self.Q = (ambient_thick * ambient_vel + turb_thick * turb_vel)

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
        """calculate time step length based on CFL condition with
           a safe rate alpha

        Return
        ---------
        dt : float
            A time step length to be used as dt_local.

        """
        dt = self.alpha * self.dx / \
            np.amax(np.array([np.amax(np.abs(self.U_link)), 1.0]))
        return dt

    def run_one_step(self, dt=None):
        """ Calculate one step. The model runs for dt seconds.
            Internally, it uses a local time step that is determined
            by CFL condition.

            Parameter
            -----------
            dt : float, optional
               Time that will elapsed by this step (s). If not specified,
               internal time step obtained by the function calc_time_step()
               is used.

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
            up_node, down_node = self.find_current_direction(
                self.U_node, self.core_nodes)
            up_link, down_link = self.find_current_direction(
                self.U_link, self.core_links)

            # Calculate advection phases of h and U respectively
            self.cip_1d_advection(
                self.h_node,
                self.dhdx,
                self.get_adv_vel(self.h_node, self.U_node, self.eta_node),
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
                self.get_adv_vel(self.h_node, self.U_node, self.eta_node),
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
                self.get_adv_vel(self.h_node, self.U_node, self.eta_node),
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
            self.cip_1d_nonadvection(
                self.h_node,
                self.dhdx,
                self.U_node,
                self.G_h,
                self.core_nodes,
                up_node,
                down_node,
                self.dx,
                dt_local,
                out_f=self.h_temp,
                out_dfdx=self.dhdx_temp)
            self.cip_1d_nonadvection(
                self.U_link,
                self.dUdx,
                self.U_link,
                self.G_U,
                self.core_links,
                up_link,
                down_link,
                self.dx,
                dt_local,
                out_f=self.U_temp,
                out_dfdx=self.dUdx_temp)
            self.cip_1d_nonadvection(
                self.C_node,
                self.dCdx,
                self.U_node,
                self.G_C,
                self.core_nodes,
                up_node,
                down_node,
                self.dx,
                dt_local,
                out_f=self.C_temp,
                out_dfdx=self.dCdx_temp)
            self.update_values()

            # increment the time step
            elapsed_time_local = elapsed_time_local + dt_local

        # increment the total elapsed time
        self.elapsed_time = self.elapsed_time + elapsed_time_local

    def get_adv_vel(self, h, U, eta, out=None):
        """ Calculate advection velocity of the layer of a turbidity current
        """
        if out is None:
            out = np.zeros(U.shape)

        H_eta = self.ambient_thick + self.turb_thick - eta
        Q = self.Q

        out = 2 * U * (H_eta - h) / H_eta \
            - 2 * h * (Q - U * h) / (H_eta * (H_eta - h)) - U

        return out

    def get_ew(self, U, h, C):
        """ calculate entrainemnt coefficient of ambient water to a turbidity
            current layer

            Parameters
            ----------
            U : ndarray
               Flow velocities of ambient water and a turbidity current.
               Row 0 is for an ambient water, and Row 1 is for a turbidity
               current.
            h : ndarray
               Flow heights of ambient water and a turbidity current. Row 0
               is an ambient water, and Row 1 is a turbidity current.
            C : ndarray
               Sediment concentration

            Returns
            ---------
            e_w : ndarray
               Entrainment coefficient of ambient water

        """
        Ri = np.zeros(U.shape)
        e_w = np.zeros(U.shape)

        U_abs = np.abs(U)
        flow_exist = np.where((h > self.h_init * 10) & (U_abs > 0))

        Ri[flow_exist] = self.R * self.g * C[flow_exist] * \
            h[flow_exist] / U_abs[flow_exist] ** 2
#        e_w[flow_exist] = 0.075 / \
#            np.sqrt(1 + 718. + Ri[flow_exist] ** 2.4)  # Parker et al. (1987)

        return e_w

    def find_current_direction(self, u, core):
        """ check flow directions and return upcurrent and downcurrent indeces

            Parameters
            -----------
            u : ndarray
                flow velocity of an ambient water and a turbidity current
                layers. Row 0 is for an ambient water, and Row 1 is for a
                turbidity current.
            core : ndarray
                Grid indeces showing "core" of the calculation domain.
                Boundary grids are excluded.

            Returns
            -----------
            up : ndarray
                Indeces of nodes or links that locate upcurrent to the core
                grids
            down : ndarray
                Indeces of nodes or links that locate downcurrent to the core
                grids

        """
        # Firstly, flow velocity is assumed to be positive everywhere
        up = core - 1
        down = core + 1

        # find the negative values in the flow velocity arrays
        negative_u_index = np.where(u[core] < 0)
        up[negative_u_index] = core[negative_u_index] + 1
        down[negative_u_index] = core[negative_u_index] - 1

        return up, down

    def update_values(self):
        """ Update values stored in grids, and process boundary conditions
            Mapping values at nodes and links each other
        """

        # copy temporal values to main variables
        self.h_node[:] = self.h_temp[:]
        self.U_link[:] = self.U_temp[:]
        self.C_node[:] = self.C_temp[:]
        self.dhdx[:] = self.dhdx_temp[:]
        self.dUdx[:] = self.dUdx_temp[:]
        self.dCdx[:] = self.dCdx_temp[:]

        # process boundary nodes
        self.h_node[-1] = self.h_node[-2]
        self.C_node[-1] = self.C_node[-2]
        self.U_link[-1] = self.U_link[-2]

        # update node and link values
        self.h_node[self.h_node < self.h_init] = self.h_init
        self.C_node[self.C_node < self.C_init] = self.C_init
        self.h_link[:] = (self.h_node[0:-1] + self.h_node[1:]) / 2.
        self.U_node[1:-1] = (self.U_link[0:-1] + self.U_link[1:]) / 2.
        self.U_node[0] = self.U_link[0]
        self.U_node[-1] = self.U_link[-1]
        self.C_link[:] = (self.C_node[0:-1] + self.C_node[1:]) / 2.

        # update values in the grid
        self.grid.h_a = self.ambient_thick + \
            self.turb_thick - self.h_node - self.eta_node
        self.grid.h_t = self.h_node
        self.grid.U_a = (self.Q - self.U_node * self.h_node) / self.grid.h_a
        self.grid.U_t = self.U_node
        self.grid.C = self.C_node

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
        h_t = self.h_node
        h_a = self.ambient_thick + self.turb_thick - self.eta_node - h_t
        U_t = self.U_node
        U_a = (self.Q - U_t * h_t) / h_a
        e_w = self.get_ew(U_t - U_a, self.h_node, self.C_node)
        core = self.core_nodes
        up = core - 1
        down = core + 1
        dx = self.dx

        # calculate non-advection terms
        out_G[core] = e_w[core] * np.abs(U_t[core] - U_a[core]) \
            - h_t[core] * (U_t[down] - U_t[up]) / (2 * dx)  # G_ht

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
        core = self.core_links
        up = core - 1
        down = core + 1
        dx = self.dx
        H = self.ambient_thick + self.turb_thick
        h_t = self.h_link[core]
        U_t = self.U_link[core]
        U_a = (self.Q - self.U_link[core] * self.h_link[core]) / \
            H - self.eta_link[core] - self.h_link[core]
        C = self.C_link[core]
        e_w = self.get_ew(U_t - U_a, h_t, C)
        eta = self.eta_link[core]
        R = self.R
        g = self.g
        Cf = self.Cf
        nu = self.nu
        Q = self.Q

        # calculate non-advection terms
        H_eta = H - eta
        H_eta_ht = H_eta - h_t
        Q_Utht = self.Q - U_t * h_t
        detadx = (self.eta_link[down] - self.eta_link[up]) / (2 * dx)
        dhdx = (self.h_link[down] - self.h_link[up]) / (2 * dx)
        out_G[core] = - (
            U_t**2 * H_eta_ht / (h_t * H_eta)
            + R * g * C * H_eta_ht / H_eta
            - 2 * U_t * Q_Utht / (H_eta * H_eta_ht)
            + Q_Utht**2 / (H_eta * H_eta_ht**2)
            - U_t ** 2 / h_t) * dhdx \
            - (R * g * C * H_eta_ht / H_eta
               + Q_Utht ** 2 / (H_eta * H_eta_ht**2)) * detadx \
            - 1 / h_t * (Cf * U_t * np.abs(U_t) * H_eta_ht / H_eta
                         - 2 * nu * (Q - U_t * H_eta) / (H_eta * H_eta_ht)
                         + e_w * U_t * np.abs((U_t * H_eta - Q) / H_eta_ht))

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
        h_t = self.h_node
        h_a = self.ambient_thick + self.turb_thick - self.eta_node - h_t
        U_t = self.U_node
        U_a = (self.Q - U_t * h_t) / h_a
        e_w = self.get_ew(U_t - U_a, self.h_node, self.C_node)
        C = self.C_node
        core = self.core_nodes

        # settling and entrainment of sediment are not implemented
        w_s = 0
        e_s = 0
        r_0 = 0

        # calculate non-advection terms
        out_G[core] = (w_s * (e_s - r_0 * C[core]) - e_w[core] * C[core] *
                       np.abs(U_t[core] - U_a[core])) / h_t[core]  # G_C

        return out_G

    def cip_1d_advection(self,
                         f,
                         dfdx,
                         u,
                         core,
                         up,
                         down,
                         dx,
                         dt,
                         out_f=None,
                         out_dfdx=None):
        """ calculate 1 step of advection phase by CIP method
        """

        if out_f is None:
            out_f = np.zeros(f.shape)
            out_f[0] = f[0]
            out_f[-1] = f[-1]
        if out_dfdx is None:
            out_dfdx = np.zeros(f.shape)
            out_dfdx[0] = dfdx[0]
            out_dfdx[-1] = dfdx[-1]

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

    def cip_1d_nonadvection(self,
                            f,
                            dfdx,
                            u,
                            G,
                            core,
                            up,
                            down,
                            dx,
                            dt,
                            out_f=None,
                            out_dfdx=None):
        """calculate 1 step of non-advection phase by CIP method
        """

        if out_f is None:
            out_f = np.zeros(f.shape)
            out_f[0] = f[0]
            out_f[-1] = f[-1]
        if out_dfdx is None:
            out_dfdx = np.zeros(dfdx.shape)
            out_dfdx[0] = dfdx[0]
            out_dfdx[-1] = dfdx[-1]

        D = -np.where(u > 0., 1.0, -1.0) * dx
        xi = -u * dt

        # non-advection term
        out_f[core] = f[core] + G[core] * dt
        out_dfdx[core] = dfdx[core] \
            + ((out_f[down] - f[down]) - (out_f[up] - f[up])) / \
            (-2 * D[core]) - dfdx[core] * (xi[down] - xi[up]) / (2 * D[core])

        return out_f, out_dfdx

    def plot(self, xlim=None, ylim=None):
        """ plot results
        """

        if xlim is None:
            xlim = [np.min(self.grid.x), np.max(self.grid.x)]

        if ylim is None:
            ylim = [np.min(self.grid.eta), np.max(self.grid.h_a) * 1.1]

        plt.cla()
        plt.plot(
            self.grid.x,
            self.grid.h_t + self.grid.eta,
            label='turbidity current',
            color='r')
        plt.plot(
            self.grid.x,
            self.grid.h_a + self.grid.h_t + self.grid.eta,
            label='water surface',
            color='b')
        plt.plot(self.grid.x, self.grid.eta, label='bed', color='g')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.text(xlim[0], ylim[1] - 20,
                 '$t = $ {:.0f} s.'.format(self.elapsed_time))
        plt.xlabel('Distance (m)')
        plt.ylabel('Elevation (m)')
        plt.legend()


if __name__ == "__main__":
    grid = Grid(number_of_grids=200, spacing=50.0)
    grid.eta = grid.x * -0.01
    tc = TwoLayerTurbidityCurrent(
        grid=grid,
        turb_vel=1.0,
        ambient_vel=-0.3,
        turb_thick=5.0,
        ambient_thick=100.0,
        concentration=0.01,
        alpha=0.001)
    steps = 1000
    for i in range(steps):
        tc.plot()
        plt.savefig('test2/tidal_flood_{:04d}'.format(i))
        tc.run_one_step(dt=10.0)
        print("", end='\r')
        print('{:.1f}% finished.'.format(i / steps * 100), end='\r')

    tc.plot()
    plt.savefig('test/tidal_flood_{:04d}'.format(i))
    plt.show()
