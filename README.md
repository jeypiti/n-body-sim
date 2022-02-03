# n-body-sim

2D N-body simulation in Python. Implements various integrators (Forward Euler, Leapfrog, PEFRL, RK8) as well as two methods to calculate the force between particles, namely the conventional direct sum approach and the Barnes-Hut algorithm.

# Quickstart

The simulation can be started as follows:

```sh
$ python3 -m pip install -r requirements.txt
$ python3 src/simulation.py
```

Initial conditions and the exact integration methods can be selected in `src/simulation.py`.
