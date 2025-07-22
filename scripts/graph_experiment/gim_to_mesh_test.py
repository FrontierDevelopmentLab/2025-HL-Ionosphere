import numpy as np 
import matplotlib.pyplot as plt

import grid_mesh_connectivity_temp
import icosahedral_mesh_temp
import typed_graph_temp

def test_grid_lat_lon_to_coordinates():

    # Intervals of 30 degrees.
    grid_latitude = np.array([-45., 0., 45])
    grid_longitude = np.array([0., 90., 180., 270.])

    inv_sqrt2 = 1 / np.sqrt(2)
    expected_coordinates = np.array([
        [[inv_sqrt2, 0., -inv_sqrt2],
         [0., inv_sqrt2, -inv_sqrt2],
         [-inv_sqrt2, 0., -inv_sqrt2],
         [0., -inv_sqrt2, -inv_sqrt2]],
        [[1., 0., 0.],
         [0., 1., 0.],
         [-1., 0., 0.],
         [0., -1., 0.]],
        [[inv_sqrt2, 0., inv_sqrt2],
         [0., inv_sqrt2, inv_sqrt2],
         [-inv_sqrt2, 0., inv_sqrt2],
         [0., -inv_sqrt2, inv_sqrt2]],
    ])

    coordinates = grid_mesh_connectivity._grid_lat_lon_to_coordinates(
        grid_latitude, grid_longitude)
    
    print(coordinates)
    print(expected_coordinates)

if __name__ == '__main__':
    print("Testing grid_lat_lon_to_coordinates...")
    test_grid_lat_lon_to_coordinates()