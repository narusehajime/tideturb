import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pickle
import ipdb
"""This is a model of a turbidity current influenced by tidal flows in
 a submarine canyon. The two-layer shallow water equation system is
 employed. The upper layer is an ambient water, and the lower layer
 is a turbidity current.

.. codeauthor:: Hajime Naruse

Example
--------
from tideturb import TwoLayerTurbidityCurrent, Grid, load_model
from matplotlib import pyplot as plt

    grid = Grid(number_of_grids=500, spacing=10.0)
    grid.eta = grid.x * -0.01
    tc = TwoLayerTurbidityCurrent(
        grid=grid,
        turb_vel=2.0,
        ambient_vel=0.3,
        turb_thick=5.0,
        ambient_thick=100.0,
        concentration=0.01,
        alpha=0.02,
        implicit_repeat_num=20,
        )
    steps = 500
    for i in range(steps):
        tc.plot()
        plt.savefig('test6/tidal_flood_{:04d}'.format(i))
        tc.run_one_step(dt=10.0)
        print("", end='\r')
        print('{:.1f}% finished.'.format(i / steps * 100), end='\r')

    tc.plot()
    plt.savefig('test6/tidal_flood_{:04d}'.format(i))
    plt.show()
    tc.save('test6_5000sec')

"""


class Grid():
    # Grid used for the TwoLayerTurbidityCurrent
    # This class store x coordinates and other flow parameters to output

    def __init__(self, number_of_grids=100, start=0.0, end=None, spacing=1.0):
        """ Constractor of the class Grid

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
            nu_t=1.0 * 10**-4,
            Ds=50 * 10**-6,
            nu=1.010 * 10**-6,
            h_init=0.0001,
            C_init=0.0001,
            h_e=0.01,
            alpha=0.01,
            implicit_repeat_num=5,
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
               Gravity acceleration. Default value is 9.81 (m/s)

            Cf : float, optional
               Bed friction coefficient. Default value is 0.004.

            nu : float, optional
               Kinematic visocity of water. Default value is 1.010*10**-6

            Ds : float, optional
               Sediment particle diamter. Default value is 50 microns.

            nu_t : float, optional
               Eddy visosity at the interface between two layers

            h_init : float, optional
               Dummy flow thickness of turbidity current. This is needed for
               numerical stability.

            h_e : float, optional
               Criterion for judging wet and dry grids

            alpha : float, optional
                Coefficient to determine the time step length considering
                Courant Number. Default is 0.01.

            implicit_repeat_number : float, optional
                Number of repetition for calculating implicit scheme. Default
                is 5.

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
            self.nu_t = nu_t
            self.Ds = Ds
            self.nu = nu
            self.h_init = h_init
            self.C_init = C_init
            self.ambient_vel = ambient_vel
            self.ambient_thick = ambient_thick
            self.turb_vel = turb_vel
            self.turb_thick = turb_thick
            self.concentration = concentration
            self.dx = self.grid.dx
            self.alpha = alpha
            self.implicit_repeat_num = implicit_repeat_num
            self.dt = 0.1
            self.elapsed_time = 0.0
            self.h_e = h_e

            # Calculate subordinate parameters
            self.ws = self.get_ws()

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
        self.h_node = np.zeros([2, self.grid.x.shape[0]])
        self.h_link = np.zeros([2, self.grid.x.shape[0] - 1])
        self.U_node = np.zeros([2, self.grid.x.shape[0]])
        self.U_link = np.zeros([2, self.grid.x.shape[0] - 1])
        self.C_node = np.zeros([2, self.grid.x.shape[0]])
        self.C_link = np.zeros([2, self.grid.x.shape[0] - 1])

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
        core_nodes = np.tile(self.grid.core_nodes, (self.h_node.shape[0], 1))
        core_links = np.tile(self.grid.core_links, (self.U_link.shape[0], 1))
        self.core_nodes = tuple(
            (np.array([np.arange(self.h_node.shape[0], dtype='int')]).T *
             np.ones(core_nodes.shape, dtype='int'), core_nodes))
        self.core_links = tuple(
            (np.array([np.arange(self.U_link.shape[0], dtype='int')]).T *
             np.ones(core_links.shape, dtype='int'), core_links))

        # Set initial and boundary conditions
        self.h_node[1, 0] = turb_thick
        # self.h_node[1, 1:] = h_init * np.ones(self.h_node[1, 1:].shape)
        self.h_node[1, 1:] = turb_thick * np.ones(self.h_node[1, 1:].shape)
        self.h_node[0, :] = ambient_thick + turb_thick - self.grid.eta - \
            self.h_node[1, :]  # initial water surface is flat
        self.h_link[0, :] = (self.h_node[0, :-1] + self.h_node[0, 1:]) / 2.
        self.U_link[0, :] = ambient_vel * ambient_thick / self.h_link[0, :]
        self.U_link[1, 0] = turb_vel
        self.U_link[1, 1:] = turb_vel * np.ones(self.U_link[1, 1:].shape)
        self.C_node[1, 0] = concentration
        # self.C_node[1, 1:] = np.ones(self.C_node.shape[1] - 1) * self.C_init
        self.C_node[1, 1:] = concentration * np.ones(self.C_node[1, 1:].shape)

        self.dhdx[:, 1:-1] = (self.h_node[:, :-2] - self.h_node[:, 2:]) / (
            2 * self.dx)
        self.dhdx[:, 0] = self.dhdx[:, 1]
        self.dhdx[:, -1] = self.dhdx[:, -2]

        # set topography
        self.eta_node = self.grid.eta
        self.eta_link = (self.eta_node[0:-1] + self.eta_node[1:]) / 2.

        # Map node and link values each other
        self.update_values(self.h_node, self.h_link, self.U_node, self.U_link,
                           self.C_node, self.C_link)

        # variables to store calculation results temporary
        self.h_node_temp = self.h_node.copy()
        self.h_link_temp = self.h_link.copy()
        self.U_node_temp = self.U_node.copy()
        self.U_link_temp = self.U_link.copy()
        self.C_node_temp = self.C_node.copy()
        self.C_link_temp = self.C_link.copy()
        self.dhdx_temp = self.dhdx.copy()
        self.dUdx_temp = self.dUdx.copy()
        self.dCdx_temp = self.dCdx.copy()

        # Store variables in the grid object
        self.copy_temp_to_main_variables()

        # Making figures
        self.fig, (self.axL, self.axM, self.axR) = plt.subplots(
            ncols=3, figsize=(25, 6))

    def calc_time_step(self):
        """calculate time step length based on CFL condition with
           a safe rate alpha

        Return
        ---------
        dt : float
            A time step length to be used as dt_local.

        """
        advel = np.abs(self.U_link) + np.sqrt(self.g * self.h_link)
        dt = self.alpha * self.dx / np.amax(advel)
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

        # set the local measurement of elapsed time and the duration
        # of this run
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

            # find wet and partial wet nodes and links
            wet_node, wet_node_up, wet_node_down = self.find_wet_grids(
                self.h_node, self.core_nodes, up_node, down_node)
            wet_link, wet_link_up, wet_link_down = self.find_wet_grids(
                self.h_link, self.core_links, up_link, down_link)

            # self.U_link[self.core_links][1][dry_link] = self.U_link[up_link][
            #     1][dry_link]
            # dry_node = self.h_node[self.core_nodes][1] < self.h_init * 10
            # self.U_node[self.core_nodes][1][dry_node] = self.U_node[up_node][
            #     1][dry_node]

            # Calculate advection phases of h, U and C
            U_flat = np.zeros(self.U_link.shape)
            U_flat[1, :] = 2 * self.U_link[1, :] * self.h_link[0, :] \
                / (self.h_link[0, :] + self.h_link[1, :]) \
                - 2 * self.h_link[1, :] * self.U_link[0, :] \
                * self.h_link[0, :] / ((self.h_link[0, :] + self.h_link[1, :])
                                       * self.h_link[0, :]) - self.U_link[1, :]
            self.rcip_1d_advection(
                self.h_node,
                self.dhdx,
                self.U_node,
                # self.core_nodes,
                wet_node,
                # up_node,
                # down_node,
                wet_node_up,
                wet_node_down,
                self.dx,
                dt_local,
                out_f=self.h_node_temp,
                out_dfdx=self.dhdx_temp)
            self.rcip_1d_advection(
                self.U_link,
                self.dUdx,
                # self.U_link,
                U_flat,
                # self.core_links,
                # up_link,
                # down_link,
                wet_link,
                wet_link_up,
                wet_link_down,
                self.dx,
                dt_local,
                out_f=self.U_link_temp,
                out_dfdx=self.dUdx_temp)
            self.rcip_1d_advection(
                self.C_node,
                self.dCdx,
                self.U_node,
                # self.core_nodes,
                wet_node,
                # up_node,
                # down_node,
                wet_node_up,
                wet_node_down,
                self.dx,
                dt_local,
                out_f=self.C_node_temp,
                out_dfdx=self.dCdx_temp)

            # Update values to store results of advection phase
            self.update_values(self.h_node_temp, self.h_link_temp,
                               self.U_node_temp, self.U_link_temp,
                               self.C_node_temp, self.C_link_temp)
            self.copy_temp_to_main_variables()

            # Calculate non-advection phase of h and U
            for i in range(self.implicit_repeat_num):

                self.calc_G_h_flat(
                    self.h_node_temp,
                    self.U_node_temp,
                    self.C_node_temp,
                    core=wet_node,
                    up=wet_node_up,
                    down=wet_node_down,
                    out_G=self.G_h)

                self.calc_G_U_flat(
                    self.h_link_temp,
                    self.U_link_temp,
                    self.C_link_temp,
                    core=wet_link,
                    up=wet_link_up,
                    down=wet_link_down,
                    out_G=self.G_U)

                self.calc_G_C(
                    self.h_node_temp,
                    self.U_node_temp,
                    self.C_node_temp,
                    core=wet_node,
                    up=wet_node_up,
                    down=wet_node_down,
                    out_G=self.G_C)

                self.cip_1d_nonadvection(
                    self.h_node,
                    self.dhdx,
                    self.U_node,
                    self.G_h,
                    # self.core_nodes,
                    wet_node,
                    # up_node,
                    # down_node,
                    wet_node_up,
                    wet_node_down,
                    self.dx,
                    dt_local,
                    out_f=self.h_node_temp,
                    out_dfdx=self.dhdx_temp)

                self.cip_1d_nonadvection(
                    self.U_link,
                    self.dUdx,
                    self.U_link,
                    self.G_U,
                    # self.core_links,
                    # up_link,
                    # down_link,
                    wet_link,
                    wet_link_up,
                    wet_link_down,
                    self.dx,
                    dt_local,
                    out_f=self.U_link_temp,
                    out_dfdx=self.dUdx_temp)

                self.cip_1d_nonadvection(
                    self.C_node,
                    self.dCdx,
                    self.U_node,
                    self.G_C,
                    # self.core_nodes,
                    # up_node,
                    # down_node,
                    wet_node,
                    wet_node_up,
                    wet_node_down,
                    self.dx,
                    dt_local,
                    out_f=self.C_node_temp,
                    out_dfdx=self.dCdx_temp)

                self.update_values(self.h_node_temp, self.h_link_temp,
                                   self.U_node_temp, self.U_link_temp,
                                   self.C_node_temp, self.C_link_temp)

            self.copy_temp_to_main_variables()

            # increment the time step
            elapsed_time_local = elapsed_time_local + dt_local

        # increment the total elapsed time
        self.elapsed_time = self.elapsed_time + elapsed_time_local

    def find_wet_grids(self, h, core, up, down):
        """Find wet nodes or links

           Parameters
           --------------------------
           h : ndarray
               flow height values for detecting wet and dry grids

           core : tuple
               core nodes or links. The first component of tuple is ndarry
               indicating the layer num (0 or 1), and the next component is
               ndarry indicating indeces of core nodes or links

            up : tuple
               tuple indicating upcurrent nodes or links.

            down : tuple
               tuple indicating downcurrent nodes or links.

            Returns
            -------------------------
            wet_grids : tuple
               tuple indicating wet grids. The format of tuple is same as core.
               Definition of wet should be determined by self.h_e value.

            wet_up : tuple
               indeces of upcurrent grids

            wet_down: tuple
               indeces of downcurrent grids
        """
        wet_boolean = ((h[core][1] > self.h_e) & (h[down][1] > self.h_e))

        wet_grids = tuple((core[0][:, wet_boolean], core[1][:, wet_boolean]))
        wet_up = tuple((up[0][:, wet_boolean], up[1][:, wet_boolean]))
        wet_down = tuple((down[0][:, wet_boolean], down[1][:, wet_boolean]))

        return wet_grids, wet_up, wet_down

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
        Ri = np.zeros(U.shape[1])
        e_w = np.zeros(U.shape[1])

        U_abs = np.abs(U[0, :] - U[1, :])
        flow_exist = np.where((h[1, :] > self.h_init * 10) & (U_abs > 0))

        Ri[flow_exist] = self.R * self.g * C[1, flow_exist] * \
            h[1, flow_exist] / U_abs[flow_exist] ** 2
        e_w[flow_exist] = 0.075 / \
            np.sqrt(1 + 718. + Ri[flow_exist] ** 2.4)  # Parker et al. (1987)

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
        up = tuple((core[0], core[1] - 1))
        down = tuple((core[0], core[1] + 1))

        # find the negative values in the flow velocity arrays
        negative_u_index = np.where(u[core] < 0)
        up[1][negative_u_index] = core[1][negative_u_index] + 1
        down[1][negative_u_index] = core[1][negative_u_index] - 1

        return up, down

    def copy_temp_to_main_variables(self):
        """Copy temporary variables to main variables, and update values
           stored in the grid object
        """

        # copy temporal values to main variables
        self.h_node[:, :] = self.h_node_temp[:, :]
        self.h_link[:, :] = self.h_link_temp[:, :]
        self.U_node[:, :] = self.U_node_temp[:, :]
        self.U_link[:, :] = self.U_link_temp[:, :]
        self.C_node[:, :] = self.C_node_temp[:, :]
        self.C_link[:, :] = self.C_link_temp[:, :]
        self.dhdx[:, :] = self.dhdx_temp[:, :]
        self.dUdx[:, :] = self.dUdx_temp[:, :]
        self.dCdx[:] = self.dCdx_temp[:]

        # update values in the grid
        self.grid.h_a = self.h_node[0, :]
        self.grid.h_t = self.h_node[1, :]
        self.grid.U_a = self.U_node[0, :]
        self.grid.U_t = self.U_node[1, :]
        self.grid.C = self.C_node[1, :]

    def update_values(self, h_node, h_link, U_node, U_link, C_node, C_link):
        """ Update values stored in grids, and process boundary conditions
            Mapping values at nodes and links each other
        """

        # boundary conditions
        ambient_thick = self.ambient_thick
        ambient_vel = self.ambient_vel
        turb_thick = self.turb_thick
        turb_vel = self.turb_vel
        concentration = self.concentration

        # process upstream boundary conditions
        h_node[0, 0] = ambient_thick
        h_node[1, 0] = turb_thick
        U_node[0, 0] = ambient_vel
        U_node[1, 0] = turb_vel
        C_node[1, 0] = concentration

        # process downstream boundary conditions
        # h_node[1, -1] = (h_node[1, -2] + h_node[1, -3] +
        #                  h_node[1, -4]) / 3.  # no gradient in t.c. layer
        h_node[1, -1] = h_node[1, -2]
        h_node[0, -1] = ambient_thick + turb_thick - \
            self.grid.eta[-1] - h_node[1, -1]  # sea surface is flat
        C_node[1, -1] = C_node[1, -2]  # no gradient in t.c. layer

        # remove negative values
        h_node[h_node < self.h_init] = self.h_init
        C_node[1, C_node[1, :] < self.C_init] = self.C_init

        # update node and link values
        h_link[:, :] = (h_node[:, :-1] + h_node[:, 1:]) / 2.
        U_node[:, 1:-1] = (U_link[:, 0:-1] + U_link[:, 1:]) / 2.
        C_link[:, :] = (C_node[:, :-1] + C_node[:, 1:]) / 2.

        # re-process boundary nodes/links to maintain constant discharge
        U_link[:, 0] = (U_node[:, 0] * h_node[:, 0]) / h_link[:, 0]
        U_link[0, -1] = U_node[0, -2] * h_node[0, -2] / h_link[0, -1]
        # U_link[1, -1] = (U_link[1, -2] + U_link[1, -3] + U_link[1, -4]) / 3.
        U_link[1, -1] = U_link[1, -2]
        U_node[0, -1] = U_link[0, -1] * h_link[0, -1] / h_node[0, -1]
        U_node[1, -1] = U_node[1, -2]

        # update values of ambient water
        self.h_node[0, :] = ambient_thick + turb_thick - self.grid.eta - \
            self.h_node[1, :]  # water surface is flat
        self.h_link[0, :] = ambient_thick + turb_thick - self.eta_link - \
            self.h_link[1, :]  # water surface is flat
        self.U_link[0, :] = ambient_vel * ambient_thick / self.h_link[0, :]
        self.U_node[0, :] = ambient_vel * ambient_thick / self.h_node[0, :]

    def update_values_flat(self, h_node, h_link, U_node, U_link, C_node,
                           C_link):
        """ Update values stored in grids, and process boundary conditions
            Mapping values at nodes and links each other
        """

        ambient_thick = self.ambient_thick
        ambient_vel = self.ambient_vel
        turb_thick = self.turb_thick
        turb_vel = self.turb_vel
        concentration = self.concentration

        # process upstream boundary conditions
        h_node[0, 0] = ambient_thick
        h_node[1, 0] = turb_thick
        U_node[0, 0] = ambient_vel
        U_node[1, 0] = turb_vel
        C_node[1, 0] = concentration

        # process downstream boundary conditions
        # h_node[1, -1] = (h_node[1, -2] + h_node[1, -3] +
        #                  h_node[1, -4]) / 3.  # no gradient in t.c. layer
        h_node[1, -1] = h_node[1, -2]
        C_node[1, -1] = C_node[1, -2]  # no gradient in t.c. layer

        # remove negative values
        h_node[h_node < self.h_init] = self.h_init
        C_node[1, C_node[1, :] < self.C_init] = self.C_init

        # update node and link values
        h_link[:, :] = (h_node[:, :-1] + h_node[:, 1:]) / 2.
        U_node[:, 1:-1] = (U_link[:, 0:-1] + U_link[:, 1:]) / 2.
        C_link[:, :] = (C_node[:, :-1] + C_node[:, 1:]) / 2.

        # update values of ambient water
        self.h_node[0, :] = ambient_thick + turb_thick - self.grid.eta - \
            self.h_node[1, :]  # water surface is flat
        self.h_link[0, :] = ambient_thick + turb_thick - self.eta_link - \
            self.h_link[1, :]  # water surface is flat
        self.U_link[0, :] = ambient_vel * ambient_thick / self.h_link[0, :]
        self.U_node[0, :] = ambient_vel * ambient_thick / self.h_node[0, :]

    def calc_G_h(self,
                 h_node,
                 U_node,
                 C_node,
                 core=None,
                 up=None,
                 down=None,
                 out_G=None):
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
        if core is None:
            core = self.core_nodes[1][0, :]
        if up is None:
            up = core - 1
        if down is None:
            down = core + 1

        # set parameters
        h_a = h_node[0, :]
        h_t = h_node[1, :]
        U_a = U_node[0, :]
        U_t = U_node[1, :]
        e_w = self.get_ew(U_node, h_node, C_node)
        icore = core[1][1, :]
        iup = icore - 1
        idown = icore + 1
        dx = self.dx

        # calculate non-advection terms
        out_G[0, icore] = - e_w[icore] * \
            np.abs(U_t[icore] - U_a[icore]) - h_a[icore] * \
            (U_a[idown] - U_a[iup]) / (2 * dx)  # G_ha
        out_G[1, icore] = e_w[icore] * np.abs(U_t[icore] - U_a[icore]) \
            - h_t[icore] * (U_t[idown] - U_t[iup]) / (2 * dx)  # G_ht

        return out_G

    def calc_G_h_flat(self,
                      h_node,
                      U_node,
                      C_node,
                      core=None,
                      up=None,
                      down=None,
                      out_G=None):
        """Calculate non-advection terms for flow heights

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
        if core is None:
            core = self.core_nodes[1][0, :]
        if up is None:
            up = core - 1
        if down is None:
            down = core + 1

        # set parameters
        h_a = h_node[0, :]
        h_t = h_node[1, :]
        U_a = U_node[0, :]
        U_t = U_node[1, :]
        e_w = self.get_ew(U_node, h_node, C_node)
        # core = self.core_nodes[1][0, :]
        icore = core[1][1, :]
        iup = icore - 1
        idown = icore + 1
        dx = self.dx
        H_minus_eta = h_a + h_t
        Q = U_a * h_a + U_t * h_t

        # calculate non-advection terms
        out_G[1, icore] = - h_t[icore] * (U_t[idown] - U_t[iup]) / (2 * dx) \
            + e_w[icore] * np.abs((U_t[icore] * H_minus_eta[icore] - Q[icore])
                                  / (H_minus_eta[icore] - h_t[icore]))

        return out_G

    def calc_G_U(self,
                 h_link,
                 U_link,
                 C_link,
                 core=None,
                 up=None,
                 down=None,
                 out_G=None):
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
        if core is None:
            core = self.core_links[1][0, :]
        if up is None:
            up = core - 1
        if down is None:
            down = core + 1

        # set parameters
        h_a = h_link[0, :]
        h_t = h_link[1, :]
        U_a = U_link[0, :]
        U_t = U_link[1, :]
        C = C_link[1, :]
        eta = self.eta_link
        e_w = self.get_ew(U_link, h_link, C_link)
        R = self.R
        g = self.g
        Cf = self.Cf
        nu_t = self.nu_t
        dx = self.dx
        icore = core[1][1, :]
        iup = icore - 1
        idown = icore + 1

        # calculate non-advection terms
        out_G[0, icore] = -g * (eta[idown] - eta[iup]) / \
            (2 * dx) - g * ((h_a[idown] + h_t[idown]) - (h_a[iup] + h_t[iup]))\
            / (2 * dx) - 2 * nu_t / \
            h_a[icore] * (U_a[icore] - U_t[icore]) / (h_a[icore] + h_t[icore]) + \
            e_w[icore] * np.abs(U_t[icore] - U_a[icore]) * \
            U_a[icore] / h_a[icore]
        out_G[1, icore] = -(1 + R * C[icore]) * g * (eta[idown] - eta[iup]) \
            / (2 * dx) - g * ((h_a[idown] + h_t[idown]) - (h_a[iup] + h_t[iup]))\
            / (2 * dx) - R * C[icore] * g * (h_t[idown] - h_t[iup]) / (
            2 * dx) + 2 * nu_t / h_t[icore] * (U_a[icore] - U_t[icore]) / \
            (h_a[icore] + h_t[icore]) - Cf * U_t[icore] * np.abs(U_t[icore]) \
            / h_t[icore] - e_w[icore] * (U_t[icore] - U_a[icore]) * U_t[icore] \
            / h_t[icore]

        return out_G

    def calc_G_U_flat(self,
                      h_link,
                      U_link,
                      C_link,
                      core=None,
                      up=None,
                      down=None,
                      out_G=None):
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
        if core is None:
            core = self.core_links

        # set parameters
        h_a = h_link[0, :]
        h_t = h_link[1, :]
        U_a = U_link[0, :]
        U_t = U_link[1, :]
        C = C_link[1, :]
        eta = self.eta_link
        e_w = self.get_ew(U_link, h_link, C_link)
        R = self.R
        g = self.g
        Cf = self.Cf
        nu_t = self.nu_t
        # core = self.core_links[1][0, :]
        icore = core[1][1, :]
        iup = icore - 1
        idown = icore + 1
        dx = self.dx
        H_minus_eta = h_a + h_t
        Q = U_a * h_a + U_t * h_t

        # calculate non-advection terms
        out_G[1, icore] = -(
            (U_t[icore]**2 * h_a[icore]) / (h_t[icore] * H_minus_eta[icore]) +
            (R * C[icore] * g * (h_a[icore])) / (H_minus_eta[icore]) -
            (2 * U_t[icore] * (Q[icore] - U_t[icore] * h_t[icore])) /
            (H_minus_eta[icore] * (h_a[icore])) + (
                (Q[icore] - U_t[icore] * h_t[icore])**2) /
            (H_minus_eta[icore] *
             (H_minus_eta[icore] - h_t[icore])**2) - (U_t[icore]**2) /
            (h_t[icore])) * (h_t[idown] - h_t[iup]) / (2 * dx) - (
                (R * C[icore] * g * h_a[icore]) / (H_minus_eta[icore]) + (
                    (Q[icore] - U_t[icore] * h_t[icore])**2) /
                (H_minus_eta[icore] * (H_minus_eta[icore] - h_t[icore])**2)
            ) * (eta[idown] - eta[iup]) / (2 * dx) - 1 / h_t[icore] * (
                (Cf * U_t[icore] * np.abs(U_t[icore]) *
                 (h_a[icore]) / H_minus_eta[icore] - 2 * nu_t *
                 (Q[icore] - U_t[icore] * H_minus_eta[icore]) /
                 (H_minus_eta[icore] * (H_minus_eta[icore] - h_t[icore])) +
                 e_w[icore] * U_t[icore] * np.abs(
                     (U_t[icore] * H_minus_eta[icore] - Q[icore]) /
                     (H_minus_eta[icore] - h_t[icore]))))
        return out_G

    def calc_G_C(self,
                 h_node,
                 U_node,
                 C_node,
                 core=None,
                 up=None,
                 down=None,
                 out_G=None):
        """ Calculate non-advection terms for concentration

            Parameter
            -----------
            h_node : ndarray
                flow height at nodes

            U_node : ndarray
                flow velocity at nodes

            C_node : ndarray
                sediment concentration at nodes

            out_G : ndarray, optional
                An array to return the calculation result

            Return
            ------------
            out_G : ndarray
                Calculation result
        """
        if out_G is None:
            out_G = np.zeros(self.G_C.shape)
        if core is None:
            core = self.core_nodes[1][0, :]

        # set parameters
        h_t = h_node[1, :]
        U_a = U_node[0, :]
        U_t = U_node[1, :]
        C = C_node[1, :]
        e_w = self.get_ew(U_node, h_node, C_node)
        ws = self.ws
        r_0 = 2.0
        icore = core[1][1, :]

        # entrainment of sediment are not implemented
        e_s = self.get_es(U_t)

        # calculate non-advection terms
        out_G[1, icore] = (
            ws * (e_s[icore] - r_0 * C[icore]) - e_w[icore] * C[icore] *
            np.abs(U_t[icore] - U_a[icore])) / h_t[icore]

        return out_G

    def get_es(self, U, es=None):
        """ Calculate entrainment rate of basal sediment to suspension
            Based on Garcia and Parker (1991)

            Parameters
            --------------
            U : ndarray
                flow velocity

            Returns
            ---------------
            es : ndarray
                dimensionless entrainment rate of basal sediment into
                suspension
        """

        if es is None:
            es = np.zeros(U.shape)

        # basic parameters
        R = self.R
        g = self.g
        Ds = self.Ds
        nu = self.nu
        ws = self.ws
        Cf = self.Cf

        # calculate subordinate parameters
        Rp = np.sqrt(R * g * Ds) * Ds / nu
        u_star = Cf * U**2
        sus_index = u_star / ws

        # coefficients for calculation
        a = 7.8 * 10**-7
        alpha = 0.6
        p = 1.0

        # calculate entrainemnt rate
        Z = sus_index * Rp**alpha
        # nonzero = Z > 0
        # es[nonzero] = p * a * Z[nonzero]**5 / (1 + (a / 0.3) * Z[nonzero]**5)
        es = p * a * Z**5 / (1 + (a / 0.3) * Z**5)

        return es

    def get_ws(self):
        """ Calculate settling velocity of sediment particles
            on the basis of Ferguson and Church (1982)

        Return
        ------------------
        ws : settling velocity of sediment particles [m/s]

        """

        # copy parameters to local variables
        R = self.R
        g = self.g
        Ds = self.Ds
        nu = self.nu

        # Coefficients for natural sands
        C_1 = 18.
        C_2 = 1.0

        ws = R * g * Ds**2 / (C_1 * nu + (0.75 * C_2 * R * g * Ds**3)**0.5)

        return ws

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
            out_f[:, 0] = f[:, 0]
            out_f[:, -1] = f[:, -1]
        if out_dfdx is None:
            out_dfdx = np.zeros(f.shape)
            out_dfdx[:, 0] = dfdx[:, 0]
            out_dfdx[:, -1] = dfdx[:, -1]

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

    def rcip_1d_advection(self,
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
        """ calculate 1 step of advection phase by rational function
            CIP method.

            Parameters
            ----------------
            f : ndarray
                variable to be calculated

            dfdx : ndarray
                spatial gradient of the parameter f

            u : ndarray
                advection velocity of f

            core : ndarray
                indeces of core grids

            up : ndarray
                indeces of grids that locate upstream

            down : ndarray
                indeces of grids that locate downstream

            dx : float
                spatial grid spacing

            dt : float
                time step length

            out_f : ndarray
                resultant value of f

            out_dfdx : ndarray
                resultant value of dfdx

            Returns
            --------------------
            out_f : ndarray
                output value of f

            out_dfdx : ndarray
                output value of dfdx

        """

        if out_f is None:
            out_f = np.zeros(f.shape)
            out_f[:, 0] = f[:, 0]
            out_f[:, -1] = f[:, -1]
        if out_dfdx is None:
            out_dfdx = np.zeros(f.shape)
            out_dfdx[:, 0] = dfdx[:, 0]
            out_dfdx[:, -1] = dfdx[:, -1]

        # advection phase
        D = -np.where(u > 0., 1.0, -1.0) * dx
        xi = -u * dt
        BB = np.ones(D[core].shape)
        alpha = np.zeros(D[core].shape)
        S = (f[up] - f[core]) / D[core]
        dz_index = (dfdx[up] - S) == 0.0
        BB[dz_index] = -1.0 / D[core][dz_index]
        BB[~dz_index] = (np.abs((S[~dz_index] - dfdx[core][~dz_index]) /
                                (dfdx[up][~dz_index] - S[~dz_index] + 1.e-10))
                         - 1.0) / D[core][~dz_index]
        alpha[(S - dfdx[core]) / (dfdx[up] - S + 1.e-10) >= 0.0] = 1.0

        a = (dfdx[core] - S +
             (dfdx[up] - S) * (1.0 + alpha * BB * D[core])) / (D[core]**2)
        b = S * alpha * BB + (S - dfdx[core]) / D[core] - a * D[core]
        c = dfdx[core] + f[core] * alpha * BB

        out_f[core] = (((a * xi[core] + b) * xi[core] + c)
                       * xi[core] + f[core]) \
            / (1.0 + alpha * BB * xi[core])
        out_dfdx[core] = ((3. * a * xi[core] + 2. * b) * xi[core] + c) \
            / (1.0 + alpha * BB * xi[core]) \
            - out_f[core] * alpha * BB / (1.0 + alpha * BB * xi[core])

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
            out_f[:, 0] = f[:, 0]
            out_f[:, -1] = f[:, -1]
        if out_dfdx is None:
            out_dfdx = np.zeros(dfdx.shape)
            out_dfdx[:, 0] = dfdx[:, 0]
            out_dfdx[:, -1] = dfdx[:, -1]

        D = -np.where(u > 0., 1.0, -1.0) * dx
        xi = -u * dt

        # non-advection term
        out_f[core] = f[core] + G[core] * dt
        out_dfdx[core] = dfdx[core] \
            + ((out_f[down] - f[down]) - (out_f[up] - f[up])) / \
            (-2 * D[core]) - dfdx[core] * (xi[down] - xi[up]) / (2 * D[core])

        return out_f, out_dfdx

    def calc_steady_condition(self):
        """ Calculate steady flow condition at the given upstream boundary
            conditions. All variables (h, U, C) are calculate.
        """
        self.update_values()

        for i in range(len(self.grid.x) - 1):
            G_h = self.calc_G_h(self.h_node[:, i], self.U_node[:, i],
                                self.C_node[:, i])
            G_U = self.calc_G_U(self.h_link[:, i], self.U_link[:, i],
                                self.C_link[:, i])
            G_C = self.calc_G_C(self.h_node[:, i], self.U_node[:, i],
                                self.C_node[:, i])

    def plot(self,
             xlim=None,
             ylim_height=None,
             ylim_velocity=None,
             ylim_concentration=None):
        """ plot results in a figure, and save it as png image.
        """

        if xlim is None:
            xlim = [np.min(self.grid.x), np.max(self.grid.x)]
        if ylim_height is None:
            ylim_height = [np.min(self.grid.eta), np.max(self.grid.h_a) * 1.1]
        if ylim_velocity is None:
            ylim_velocity = [
                np.min([self.grid.U_t, self.grid.U_a]) * 0.9,
                np.max([self.grid.U_t, self.grid.U_a]) * 1.1
            ]
        if ylim_concentration is None:
            ylim_concentration = [0.00, 5.0]

        # clear figures
        axL = self.axL
        axM = self.axM
        axR = self.axR
        axL.cla()
        axM.cla()
        axR.cla()

        # plot flow height in left figure
        axL.plot(
            self.grid.x,
            self.grid.h_t + self.grid.eta,
            label='turbidity current',
            color='r')
        axL.plot(
            self.grid.x,
            self.grid.h_a + self.grid.h_t + self.grid.eta,
            label='water surface',
            color='b')
        axL.plot(self.grid.x, self.grid.eta, label='bed', color='g')
        axL.set_xlim(xlim[0], xlim[1])
        axL.set_ylim(ylim_height[0], ylim_height[1])
        axL.text(xlim[0], ylim_height[1] - 20,
                 '$t = $ {:.0f} s.'.format(self.elapsed_time))
        axL.set_xlabel('Distance (m)')
        axL.set_ylabel('Elevation (m)')
        axL.legend()

        # plot flow velocity in the middle figure
        axM.plot(
            self.grid.x, self.grid.U_t, label='turbidity current', color='r')
        axM.plot(self.grid.x, self.grid.U_a, label='ambient water', color='b')
        axM.set_xlim(xlim[0], xlim[1])
        axM.set_ylim(ylim_velocity[0], ylim_velocity[1])
        axM.set_xlabel('Distance (m)')
        axM.set_ylabel('Velocity (m/s)')
        axM.legend()

        # plot concentration in the right figure
        axR.plot(
            self.grid.x, self.grid.C * 100, label='concentration', color='r')
        axR.set_xlim(xlim[0], xlim[1])
        axR.set_ylim(ylim_concentration[0], ylim_concentration[1])
        axR.set_xlabel('Distance (m)')
        axR.set_ylabel('Concentration (%)')

    def save(self, filename):
        """save the result as a pickled binary data of
           the object TwoLayerTurbidityCurrent
        """

        with open(filename, 'wb') as f:
            pickle.dump(self, f)


def load_model(filename):
    """load the object TwoLayerTurbidityCurrent
    """

    with open(filename, 'rb') as f:
        tc = pickle.load(f)
        tc.fig, (tc.axL, tc.axM, tc.axR) = plt.subplots(
            ncols=3, figsize=(25, 6))
        return tc

    return None


if __name__ == "__main__":
    grid = Grid(number_of_grids=100, spacing=50.0)
    grid.eta = grid.x * -0.05
    tc = TwoLayerTurbidityCurrent(
        grid=grid,
        turb_vel=2.0,
        ambient_vel=-0.3,
        turb_thick=5.0,
        ambient_thick=100.0,
        Ds=80 * 10**-6,
        concentration=0.01,
        alpha=0.5,
        implicit_repeat_num=10,
        nu_t=0.01,
        h_init=0.001,
    )
    steps = 200
    for i in range(steps):
        tc.plot(ylim_velocity=[-0.5, 3.0])
        plt.savefig('test03/tidal_flood_{:04d}'.format(i))
        tc.run_one_step(dt=10.0)
        print("", end='\r')
        print('{:.1f}% finished.'.format(i / steps * 100), end='\r')

    tc.plot()
    plt.savefig('test03/tidal_flood_{:04d}'.format(i))
    tc.save('test03_5000sec')
    Fr = tc.U_node[1, :] / \
        np.sqrt(tc.h_node[1, :] * 1.65 * tc.C_node[1, :] * 9.81)
    print(Fr)
