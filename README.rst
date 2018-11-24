TIDETURB
========================

This is a code for modeling of tide-influenced turbidity currents. Two layer
shallow water equations are adopted in this model.

---------------
Installation

python setup.py install

---------------

Usage
---------
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
