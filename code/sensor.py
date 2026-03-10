import numpy as np
from numba import jit

@jit(nopython=True)
def get_sensor(choice):
    if choice == 1:
        dx = 16e-3
        dy = 16e-3
        num_x_pix = 512
        num_y_pix = 512
        sx = num_x_pix*dx # sensor size
        sy = num_y_pix*dy
    return sx, sy, num_x_pix, num_y_pix, dx, dy

@jit(nopython=True)
def get_grid_coords(sx, num_x_pix, sy, num_y_pix):
    # x increasing from left to right across sensor (low idx to high idx)
    # order x-coords from low to high
    # y increasing from bottom to top across sensor (y_cord increasing from large row idx to small row idx)
    # negative y coord correspond to large row idx toward bottom of sensor
    # order y-coords from high to low
    x_grid_sensor = np.linspace(0, sx, num_x_pix); y_grid_sensor = np.linspace(0, sy, num_y_pix)
    # center the pixel locations in the x-y plane
    x_grid_sensor = x_grid_sensor - x_grid_sensor[num_x_pix//2]
    y_grid_sensor = (y_grid_sensor - y_grid_sensor[num_y_pix//2])[::-1]
    return x_grid_sensor, y_grid_sensor

@jit(nopython=True)
def convert_index_to_coord(row_idx, col_idx, sensor_choice):
    row_idx = int(row_idx)
    col_idx = int(col_idx)
    sx, sy, num_x_pix, num_y_pix, dx, dy = get_sensor(sensor_choice)
    x_coords, y_coords = get_grid_coords(sx, num_x_pix, sy, num_y_pix)
    return x_coords[col_idx], y_coords[row_idx]

@jit(nopython=True)
def convert_coord_to_idx(x_coord, y_coord, sensor_choice):
    sx, sy, num_x_pix, num_y_pix, dx, dy = get_sensor(sensor_choice)
    x_coords, y_coords = get_grid_coords(sx, num_x_pix, sy, num_y_pix)
    row_idx = np.argmin(np.abs(y_coords - y_coord))
    col_idx = np.argmin(np.abs(x_coords - x_coord))
    return row_idx, col_idx

# input photon_locs_ra must be inds of photon locs as floats
@jit(nopython=True)
def get_photon_locs(photon_locs_ra, sensor_choice):
    assert(photon_locs_ra.shape[1] == 2)
    for j in range(photon_locs_ra.shape[0]):
        x, y = convert_index_to_coord(photon_locs_ra[j,0], photon_locs_ra[j,1], sensor_choice)
        photon_locs_ra[j,0] = x
        photon_locs_ra[j,1] = y
    return photon_locs_ra

