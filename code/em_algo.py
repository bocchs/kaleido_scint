import numpy as np
from numba import jit
import sensor


"""
coeffs are coefficients for posx, posy, negx, negy mirrors
loc = (x0,y0,z0)
n: index of refraction
"""
@jit(nopython=True)
def compute_derivatives(coeffs, n, h, k, loc, z_lens, S1, S2, A, a, num_clusters, min_sigma):
    if k == num_clusters - 1:
        xk_coeffs = np.array([1.,0,0])
        yk_coeffs = np.array([0,1.,0])
        zk_coeffs = np.array([0,0,1.])
        xk = np.dot(loc, xk_coeffs)
        yk = np.dot(loc, yk_coeffs)
        zk = z_lens - (h - (h-np.dot(loc, zk_coeffs))/n)
    elif k < num_clusters - 1:
        xk_coeffs = coeffs[3*k]
        yk_coeffs = coeffs[3*k+1]
        zk_coeffs = coeffs[3*k+2]
        xk = np.dot(loc, xk_coeffs)
        yk = np.dot(loc, yk_coeffs)
        zk = z_lens - (h - (h-np.dot(loc, zk_coeffs))/n)
    # d/dx0(xk/zk)
    xk_over_zk_dx0 = (xk_coeffs[0]*zk - xk*(-zk_coeffs[0]/n)) / zk**2
    xk_over_zk_dy0 = (xk_coeffs[1]*zk - xk*(-zk_coeffs[1]/n)) / zk**2
    xk_over_zk_dz0 = (xk_coeffs[2]*zk - xk*(-zk_coeffs[2]/n)) / zk**2
    yk_over_zk_dx0 = (yk_coeffs[0]*zk - yk*(-zk_coeffs[0]/n)) / zk**2
    yk_over_zk_dy0 = (yk_coeffs[1]*zk - yk*(-zk_coeffs[1]/n)) / zk**2
    yk_over_zk_dz0 = (yk_coeffs[2]*zk - yk*(-zk_coeffs[2]/n)) / zk**2

    xk_over_zk_squared_dx0 = 2*xk/zk * xk_over_zk_dx0
    xk_over_zk_squared_dy0 = 2*xk/zk * xk_over_zk_dy0
    xk_over_zk_squared_dz0 = 2*xk/zk * xk_over_zk_dz0
    yk_over_zk_squared_dx0 = 2*yk/zk * yk_over_zk_dx0
    yk_over_zk_squared_dy0 = 2*yk/zk * yk_over_zk_dy0
    yk_over_zk_squared_dz0 = 2*yk/zk * yk_over_zk_dz0

    sigmak = a*A*S2/S1*np.abs(S1-zk)/zk 
    if sigmak < min_sigma:
        sigmak = min_sigma
        sigmak_dx0 = 0
        sigmak_dy0 = 0
        sigmak_dz0 = 0
    else:
        # temp = |S1-zk|/zk
        num = S1-zk
        den = zk
        temp_dx0 = (zk_coeffs[0]*np.sign(num)*den/n - np.abs(num)*(-zk_coeffs[0]/n)) / den**2
        temp_dy0 = (zk_coeffs[1]*np.sign(num)*den/n - np.abs(num)*(-zk_coeffs[1]/n)) / den**2
        temp_dz0 = (zk_coeffs[2]*np.sign(num)*den/n - np.abs(num)*(-zk_coeffs[2]/n)) / den**2
        sigmak_dx0 = a*A*S2/S1*temp_dx0
        sigmak_dy0 = a*A*S2/S1*temp_dy0
        sigmak_dz0 = a*A*S2/S1*temp_dz0
    return xk_over_zk_dx0, xk_over_zk_dy0, xk_over_zk_dz0, yk_over_zk_dx0, yk_over_zk_dy0, yk_over_zk_dz0, xk_over_zk_squared_dx0, xk_over_zk_squared_dy0, xk_over_zk_squared_dz0, yk_over_zk_squared_dx0, yk_over_zk_squared_dy0, yk_over_zk_squared_dz0, sigmak_dx0, sigmak_dy0, sigmak_dz0


# h: pyramid height
# n: scintillator index of refraction
@jit(nopython=True)
def run_em_iters(photon_locs_ra, photon_weights, h, n, coeffs, posx_posy, negx_posy, negx_negy, posx_negy, A, S1, S2, lens_z, focal_z_apparent, a, sensor_x_grid, sensor_y_grid, sensor_choice, num_clusters, est_loc, pi, num_steps, min_sigma, compute_init_Q, use_posx, use_posy, use_negx, use_negy, posx_init, posy_init, negx_init, negy_init, center_init, lam, step_size, grad_asc_iters):
    sensor_z = lens_z + S2

    N = photon_locs_ra.shape[0]
    locs_steps_ra = np.zeros((num_steps+1, 3))
    locs_steps_ra[0] = est_loc
    mirrors_ra = ["posx","posy","negx","negy","none"]

    # order of images in coeffs: posx, posy, negx, negy, direct
    # E step: compute new r_ik using new parameter values (x,y,z)
    r_ra = np.zeros((N,num_clusters)) # reponsibilities array
    mu_ra = np.zeros((num_clusters,2))
    sigma_ra = np.zeros((num_clusters,))
    exp_term = np.zeros((N, num_clusters))
    for k in range(num_clusters):
        if (k == 0 and not use_posx) or (k == 1 and not use_posy) or (k == 2 and not use_negx) or (k == 3 and not use_negy):
            continue
        if k == num_clusters-1:
            # corresponds to direct image
            x_k = est_loc[0]
            y_k = est_loc[1]
            z_wk_real = est_loc[2] # world coords
            z_ck_real = lens_z - z_wk_real
            z_wk_app = h - (h - z_wk_real) / n # correct true to apparent coords
            z_ck_app = lens_z - z_wk_app # origin at lens (camera coords), apparent coords
        elif k < num_clusters-1:
            # corresponds to mirror image
            x_k = np.dot(coeffs[3*k], est_loc)
            y_k = np.dot(coeffs[3*k+1], est_loc)
            z_wk_real = np.dot(coeffs[3*k+2], est_loc) # world coords
            z_ck_real = lens_z - z_wk_real
            z_wk_app = h - (h - z_wk_real) / n # correct true to apparent coords
            z_ck_app = lens_z - z_wk_app # origin at lens (camera coords), apparent coords
        if z_ck_app == S1 or z_ck_app == 0:
            continue
        else:
            sigma_k = a*A*S2/S1*np.abs(S1-z_ck_app)/z_ck_app 
            if sigma_k < min_sigma:
                sigma_k = min_sigma
        mu_k = S2/z_ck_app*np.array([x_k, y_k])
        mu_ra[k] = mu_k
        sigma_ra[k] = sigma_k
        mirror_name = mirrors_ra[k]
        mirror_image_loc_real = np.array([x_k, y_k, z_wk_real]) # origin at scintillator bottom
        mirror_image_loc_app = np.array([x_k, y_k, z_wk_app])
        for i in range(N):
            t_i = photon_locs_ra[i]
            exp_term[i,k] = -1/(2*sigma_ra[k]**2)*np.dot(t_i - mu_ra[k], t_i - mu_ra[k])
            r, c = sensor.convert_coord_to_idx(t_i[0], t_i[1], sensor_choice)
    for i in range(N):
        t_i = photon_locs_ra[i]
        denom = 0
        for k in range(num_clusters):
            if (k == 0 and not use_posx) or (k == 1 and not use_posy) or (k == 2 and not use_negx) or (k == 3 and not use_negy):
                continue
            denom += pi[k] * 1/(2*np.pi*sigma_ra[k]**2) * np.exp(exp_term[i,k])
        if denom == 0:
            r_ra[i] = 0
        else:
            for k in range(num_clusters):
                if (k == 0 and not use_posx) or (k == 1 and not use_posy) or (k == 2 and not use_negx) or (k == 3 and not use_negy):
                    continue
                r_ik = (pi[k] * 1/(2*np.pi*sigma_ra[k]**2) * np.exp(exp_term[i,k])) / denom
                r_ra[i,k] = r_ik


    # M step: compute new parameter values (x,y,z) using new r_ik by maximizing Q
    if not compute_init_Q:
        # only update est_loc when actually running EM algo
        for stp in range(grad_asc_iters):
            dQdx0 = 0.
            dQdy0 = 0.
            dQdz0 = 0.
            # compute gradient
            for i in range(N):
                ci = photon_weights[i]
                ti = photon_locs_ra[i]
                tix = ti[0]
                tiy = ti[1]
                for k in range(num_clusters):
                    if (k == 0 and not use_posx) or (k == 1 and not use_posy) or (k == 2 and not use_negx) or (k == 3 and not use_negy):
                        continue
                    if k == num_clusters-1:
                        xk = est_loc[0]
                        yk = est_loc[1]
                        z_wk_real = est_loc[2] # world coords
                        z_ck_real = lens_z - z_wk_real
                        z_wk_app = h - (h - z_wk_real) / n # correct true to apparent coords
                        z_ck_app = lens_z - z_wk_app # origin at lens (camera coords), apparent coords
                    else:
                        xk = np.dot(coeffs[3*k], est_loc)
                        yk = np.dot(coeffs[3*k+1], est_loc)
                        z_wk_real = np.dot(coeffs[3*k+2], est_loc) # world coords
                        z_ck_real = lens_z - z_wk_real
                        z_wk_app = h - (h - z_wk_real) / n # correct true to apparent coords
                        z_ck_app = lens_z - z_wk_app # origin at lens (camera coords), apparent coords
                    zk = z_ck_app
                    sigmak = sigma_ra[k]
                    ret = compute_derivatives(coeffs, n, h, k, est_loc, lens_z, S1, S2, A, a, num_clusters, min_sigma)
                    xk_over_zk_dx0, xk_over_zk_dy0, xk_over_zk_dz0, yk_over_zk_dx0, yk_over_zk_dy0, yk_over_zk_dz0, xk_over_zk_squared_dx0, xk_over_zk_squared_dy0, xk_over_zk_squared_dz0, yk_over_zk_squared_dx0, yk_over_zk_squared_dy0, yk_over_zk_squared_dz0, sigmak_dx0, sigmak_dy0, sigmak_dz0 = ret
                    term1 = tix**2 - 2*tix*S2*xk/zk + S2**2*(xk/zk)**2 + tiy**2 - 2*tiy*S2*yk/zk + S2**2*(yk/zk)**2
                    term2_x = -2*tix*S2*xk_over_zk_dx0 + S2**2*xk_over_zk_squared_dx0 - 2*tiy*S2*yk_over_zk_dx0 + S2**2*yk_over_zk_squared_dx0
                    term2_y = -2*tix*S2*xk_over_zk_dy0 + S2**2*xk_over_zk_squared_dy0 - 2*tiy*S2*yk_over_zk_dy0 + S2**2*yk_over_zk_squared_dy0
                    term2_z = -2*tix*S2*xk_over_zk_dz0 + S2**2*xk_over_zk_squared_dz0 - 2*tiy*S2*yk_over_zk_dz0 + S2**2*yk_over_zk_squared_dz0
                    dQdx0 += r_ra[i,k] * (-2/sigmak*sigmak_dx0 - ci * (-sigmak**-3*sigmak_dx0*term1 + 1/2*sigmak**-2*term2_x))
                    dQdy0 += r_ra[i,k] * (-2/sigmak*sigmak_dy0 - ci * (-sigmak**-3*sigmak_dy0*term1 + 1/2*sigmak**-2*term2_y))
                    dQdz0 += r_ra[i,k] * (-2/sigmak*sigmak_dz0 - ci * (-sigmak**-3*sigmak_dz0*term1 + 1/2*sigmak**-2*term2_z))
                    # regularization term:
                    if use_posx and k == 0:
                        g = ((S2*xk/zk)-posx_init[0])**2 + ((S2*yk/zk)-posx_init[1])**2
                        dgdx0 = 2*(S2*xk/zk - posx_init[0])*(S2*xk_over_zk_dx0) + 2*(S2*yk/zk - posx_init[1])*(S2*yk_over_zk_dx0)
                        dgdy0 = 2*(S2*xk/zk - posx_init[0])*(S2*xk_over_zk_dy0) + 2*(S2*yk/zk - posx_init[1])*(S2*yk_over_zk_dy0)
                        dgdz0 = 2*(S2*xk/zk - posx_init[0])*(S2*xk_over_zk_dz0) + 2*(S2*yk/zk - posx_init[1])*(S2*yk_over_zk_dz0)
                    if use_posy and k == 1:
                        g = ((S2*xk/zk)-posy_init[0])**2 + ((S2*yk/zk)-posy_init[1])**2
                        dgdx0 = 2*(S2*xk/zk - posy_init[0])*(S2*xk_over_zk_dx0) + 2*(S2*yk/zk - posy_init[1])*(S2*yk_over_zk_dx0)
                        dgdy0 = 2*(S2*xk/zk - posy_init[0])*(S2*xk_over_zk_dy0) + 2*(S2*yk/zk - posy_init[1])*(S2*yk_over_zk_dy0)
                        dgdz0 = 2*(S2*xk/zk - posy_init[0])*(S2*xk_over_zk_dz0) + 2*(S2*yk/zk - posy_init[1])*(S2*yk_over_zk_dz0)
                    if use_negx and k == 2:
                        g = ((S2*xk/zk)-negx_init[0])**2 + ((S2*yk/zk)-negx_init[1])**2
                        dgdx0 = 2*(S2*xk/zk - negx_init[0])*(S2*xk_over_zk_dx0) + 2*(S2*yk/zk - negx_init[1])*(S2*yk_over_zk_dx0)
                        dgdy0 = 2*(S2*xk/zk - negx_init[0])*(S2*xk_over_zk_dy0) + 2*(S2*yk/zk - negx_init[1])*(S2*yk_over_zk_dy0)
                        dgdz0 = 2*(S2*xk/zk - negx_init[0])*(S2*xk_over_zk_dz0) + 2*(S2*yk/zk - negx_init[1])*(S2*yk_over_zk_dz0)
                    if use_negy and k == 3:
                        g = ((S2*xk/zk)-negy_init[0])**2 + ((S2*yk/zk)-negy_init[1])**2
                        dgdx0 = 2*(S2*xk/zk - negy_init[0])*(S2*xk_over_zk_dx0) + 2*(S2*yk/zk - negy_init[1])*(S2*yk_over_zk_dx0)
                        dgdy0 = 2*(S2*xk/zk - negy_init[0])*(S2*xk_over_zk_dy0) + 2*(S2*yk/zk - negy_init[1])*(S2*yk_over_zk_dy0)
                        dgdz0 = 2*(S2*xk/zk - negy_init[0])*(S2*xk_over_zk_dz0) + 2*(S2*yk/zk - negy_init[1])*(S2*yk_over_zk_dz0)
                    if k == 4:
                        g = ((S2*xk/zk)-center_init[0])**2 + ((S2*yk/zk)-center_init[1])**2
                        dgdx0 = 2*(S2*xk/zk - center_init[0])*(S2*xk_over_zk_dx0) + 2*(S2*yk/zk - center_init[1])*(S2*yk_over_zk_dx0)
                        dgdy0 = 2*(S2*xk/zk - center_init[0])*(S2*xk_over_zk_dy0) + 2*(S2*yk/zk - center_init[1])*(S2*yk_over_zk_dy0)
                        dgdz0 = 2*(S2*xk/zk - center_init[0])*(S2*xk_over_zk_dz0) + 2*(S2*yk/zk - center_init[1])*(S2*yk_over_zk_dz0)
                    dQdx0 += -lam*dgdx0
                    dQdy0 += -lam*dgdy0
                    dQdz0 += -lam*dgdz0
            grad = np.array([dQdx0, dQdy0, dQdz0])
            est_loc_update = est_loc + step_size*grad # one step in gradient ascent, repeat many steps to optimize Q
            step_dist = np.linalg.norm(est_loc_update - est_loc)
            est_loc = est_loc_update
    pi = r_ra.sum(axis=0) / N
    # compute new Q after arriving at next est_loc from optimizing Q
    Q = 0
    for i in range(N):
        ci = photon_weights[i]
        ti = photon_locs_ra[i]
        tix = ti[0]
        tiy = ti[1]
        for k in range(num_clusters):
            if (k == 0 and not use_posx) or (k == 1 and not use_posy) or (k == 2 and not use_negx) or (k == 3 and not use_negy):
                continue
            if k == num_clusters-1:
                xk = est_loc[0]
                yk = est_loc[1]
                z_wk_real = est_loc[2] # world coords
                z_ck_real = lens_z - z_wk_real
                z_wk_app = h - (h - z_wk_real) / n # correct true to apparent coords
                z_ck_app = lens_z - z_wk_app # origin at lens (camera coords), apparent coords
            else:
                xk = np.dot(coeffs[3*k], est_loc)
                yk = np.dot(coeffs[3*k+1], est_loc)
                z_wk_real = np.dot(coeffs[3*k+2], est_loc) # world coords
                z_ck_real = lens_z - z_wk_real
                z_wk_app = h - (h - z_wk_real) / n # correct true to apparent coords
                z_ck_app = lens_z - z_wk_app # origin at lens (camera coords), apparent coords
            zk = z_ck_app
            term = tix**2 - 2*tix*S2*xk/zk + S2**2*(xk/zk)**2 + tiy**2 - 2*tiy*S2*yk/zk + S2**2*(yk/zk)**2
            if r_ra[i,k] > 0:
                Q += r_ra[i,k] * (np.log(pi[k]) + np.log(ci) - np.log(2*np.pi*sigma_ra[k]**2) - ci*1/2*sigma_ra[k]**-2 * term)
    for k in range(num_clusters):
        if (k == 0 and not use_posx) or (k == 1 and not use_posy) or (k == 2 and not use_negx) or (k == 3 and not use_negy):
            continue
        if k == 0 and use_posx:
            mu_init = posx_init
        if k == 1 and use_posy:
            mu_init = posy_init
        if k == 2 and use_negx:
            mu_init = negx_init
        if k == 3 and use_negy:
            mu_init = negy_init
        if k == 4:
            mu_init = center_init
        mu = mu_ra[k]
        Q += -lam * np.linalg.norm(mu - mu_init)**2 # regularization term
    return est_loc, pi, Q, r_ra


def run_em(pyramid, photon_locs_ra, photon_weights, coeffs, posx_posy, negx_posy, negx_negy, posx_negy, est_loc, params, sensor_choice, min_sigma, compute_init_Q, use_posx, use_posy, use_negx, use_negy, posx_init, posy_init, negx_init, negy_init, center_init):
    if compute_init_Q:
        num_steps = 1
    else:
        num_steps = params["max_em_steps"]
    sx, sy, num_x_pix, num_y_pix, dx, dy = sensor.get_sensor(sensor_choice)
    sensor_x_grid_ra, sensor_y_grid_ra = sensor.get_grid_coords(sx, num_x_pix, sy, num_y_pix)
    sensor_x_grid, sensor_y_grid = np.meshgrid(sensor_x_grid_ra, sensor_y_grid_ra)
    N = photon_locs_ra.shape[0]
    num_clusters = 5
    pi = np.ones((num_clusters,)) / num_clusters
    locs_steps_ra = np.zeros((num_steps+1, 3))
    locs_steps_ra[0] = est_loc
    all_r_ra = np.zeros((num_steps+1, photon_locs_ra.shape[0], num_clusters))
    Q_ra = -np.inf*np.ones((num_steps+1))
    a = params["a"]
    A = params["lens_diam"]
    S1 = params["S1"]
    S2 = params["S2"]
    lens_z = params["lens_z"]
    focal_z_apparent = params["focal_z_apparent"]
    a = params["a"]
    step_size = params["step_size"]
    lam = params["lam"]
    nu = params["nu"]
    num_nbrs = params["num_nbrs"]
    grad_asc_iters = params["grad_asc_iters"]

    temp, temp2, Q, r_ra = run_em_iters(photon_locs_ra, photon_weights, pyramid.top_z, pyramid.n, coeffs, posx_posy, negx_posy, negx_negy, posx_negy, A, S1, S2, lens_z, focal_z_apparent, a, sensor_x_grid, sensor_y_grid, sensor_choice, num_clusters, est_loc, pi, num_steps, min_sigma, compute_init_Q, use_posx, use_posy, use_negx, use_negy, posx_init, posy_init, negx_init, negy_init, center_init, lam, step_size, grad_asc_iters)
    Q_ra[0] = Q # get Q value of initialization point

    start_loc = est_loc
    for step in range(num_steps):
        est_loc, pi, Q, r_ra = run_em_iters(photon_locs_ra, photon_weights, pyramid.top_z, pyramid.n, coeffs, posx_posy, negx_posy, negx_negy, posx_negy, A, S1, S2, lens_z, focal_z_apparent, a, sensor_x_grid, sensor_y_grid, sensor_choice, num_clusters, est_loc, pi, num_steps, min_sigma, compute_init_Q, use_posx, use_posy, use_negx, use_negy, posx_init, posy_init, negx_init, negy_init, center_init, lam, step_size, grad_asc_iters)
        locs_steps_ra[step+1] = est_loc
        Q_ra[step+1] = Q
        all_r_ra[step+1] = r_ra
        step_dist = np.linalg.norm(locs_steps_ra[step+1] - locs_steps_ra[step])
        if step_dist < .01 or np.isnan(est_loc[0]):
            end_loc = locs_steps_ra[:step+2]
            return locs_steps_ra[:step+2], Q_ra[:step+2], all_r_ra[:step+2]
    return locs_steps_ra, Q_ra, all_r_ra

