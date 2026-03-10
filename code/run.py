import numpy as np
import matplotlib.pyplot as plt
import sensor
import em_algo
from pyramid import *
import os
from itertools import combinations
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def init_values():
    sensor_choice = 1
    sx, sy, num_x_pix, num_y_pix, dx, dy = sensor.get_sensor(sensor_choice)
    pixel_pitch = dx
    min_sigma = 10 * pixel_pitch
    lam = 10 # regularization coeff
    max_em_steps = 100
    step_size = 1e-7
    grad_asc_iters = 1000
    nu = 10*pixel_pitch
    num_nbrs = 10

    A = 41.7
    S1 = 45
    S2 = 72
    a = 0.25

    opening_angle = 120
    pyramid_base_len = 20
    pyramid_height = np.tan(np.pi/2-opening_angle/2*np.pi/180) * pyramid_base_len/2
    flat_vertex_z = -1000 # for making a flat bottom surface instead of vertex corner point
    index_of_refraction = 1.91
    pyramid = Pyramid(pyramid_height, pyramid_base_len, flat_vertex_z, index_of_refraction)

    focal_z_apparent = pyramid.top_z - pyramid.top_z / pyramid.n

    lens_z = S1 + focal_z_apparent

    params = {}
    params["lens_diam"] = A
    params["lens_z"] = lens_z
    params["S1"] = S1
    params["S2"] = S2
    params["a"] = a
    params["focal_z_apparent"] = focal_z_apparent
    params["sensor_choice"] = sensor_choice
    params["pixel_pitch"] = pixel_pitch
    params["min_sigma"] = min_sigma
    params["lam"] = lam
    params["max_em_steps"] = max_em_steps
    params["step_size"] = step_size
    params["grad_asc_iters"] = grad_asc_iters
    params["nu"] = nu
    params["num_nbrs"] = num_nbrs
    return params, pyramid


def get_photon_weights(photon_locs_ra, num_nbrs=10, nu=1):
    photon_weights = np.zeros((photon_locs_ra.shape[0],))
    nbrs = NearestNeighbors(n_neighbors=num_nbrs, algorithm='ball_tree').fit(photon_locs_ra)
    distances, indices = nbrs.kneighbors(photon_locs_ra)
    distances = distances[:,1:] # don't include self distance of 0
    weights = np.sum(np.exp(-distances**2 / nu), axis=1)
    return weights


def get_centroids(photon_locs_ra, photon_weights, sensor_choice, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(photon_locs_ra, sample_weight=photon_weights)
    centroids_mm = kmeans.cluster_centers_
    centroids_pix = np.zeros_like(centroids_mm)
    for i in range(centroids_mm.shape[0]):
        x_coord = centroids_mm[i,0]
        y_coord = centroids_mm[i,1]
        row_idx, col_idx = sensor.convert_coord_to_idx(x_coord, y_coord, sensor_choice)
        centroids_pix[i,0] = row_idx
        centroids_pix[i,1] = col_idx
    return centroids_mm, centroids_pix


def get_init_locs(centroids_mm, pyramid, lens_z, S2):
    centroids_mm = centroids_mm.reshape(-1,2)
    init_loc_ra = []
    init_z_vals = np.linspace(.1, pyramid.top_z, 10)
    for i in range(centroids_mm.shape[0]):
        mu_x = centroids_mm[i,0]
        mu_y = centroids_mm[i,1]
        for init_z in init_z_vals:
            app_z = pyramid.top_z - (pyramid.top_z - init_z) / pyramid.n
            init_x = mu_x * (lens_z-app_z) / S2
            init_y = mu_y * (lens_z-app_z) / S2
            init_loc = np.array([[init_x, init_y, init_z]])
            if point_in_pyramid(init_loc[0], pyramid):
                init_loc_ra.append(init_loc)
    init_loc_ra = np.concatenate(init_loc_ra, axis=0)
    return init_loc_ra


def find_best_aligned_triplet_from_n(points, n):
    best_triplet = None
    best_score = np.inf
    best_axis = None
    excluded_points = None
    for combo in combinations(range(n), 3):
        triplet = points[list(combo)]
        remaining_index = list(set(range(n)) - set(combo))
        remainder = points[remaining_index]
        std_x = np.std(triplet[:, 0])
        std_y = np.std(triplet[:, 1])
        score = min(std_x, std_y)
        axis = 'x' if std_y < std_x else 'y'
        if score < best_score:
            best_score = score
            best_triplet = triplet
            best_axis = axis
            excluded_points = remainder
    return best_triplet, excluded_points, best_axis, best_score


def classify_3_points(points):
    roles = {}
    best_double = None
    best_score = np.inf
    best_axis = None
    excluded_point = None
    for combo in combinations(range(3), 2):
        double = points[list(combo)]
        remaining_index = list(set(range(3)) - set(combo))
        remainder = points[remaining_index]
        std_x = np.std(double[:, 0])
        std_y = np.std(double[:, 1])
        score = min(std_x, std_y)
        axis = 'x' if std_y < std_x else 'y'
        if score < best_score:
            best_score = score
            best_double = double
            best_axis = axis
            excluded_points = remainder
    if best_axis == "x":
        x_locs = best_double[:,0]
        ind_low = np.argmin(x_locs)
        ind_hi = np.argmax(x_locs)
        if abs(excluded_points[0,0] - x_locs[ind_low]) < abs(excluded_points[0,0] - x_locs[ind_hi]):
            roles["center"] = best_double[ind_low]
            roles["right"] = best_double[ind_hi]
        else:
            roles["center"] = best_double[ind_hi]
            roles["left"] = best_double[ind_low]
        if excluded_points[0,1] < roles["center"][1]:
            roles["bottom"] = excluded_points[0]
        else:
            roles["top"] = excluded_points[0]
    else:
        y_locs = best_double[:,1]
        ind_low = np.argmin(y_locs)
        ind_hi = np.argmax(y_locs)
        if abs(excluded_points[0,1] - y_locs[ind_low]) < abs(excluded_points[0,1] - y_locs[ind_hi]):
            roles["center"] = best_double[ind_low]
            roles["top"] = best_double[ind_hi]
        else:
            roles["center"] = best_double[ind_hi]
            roles["bottom"] = best_double[ind_low]
        if excluded_points[0,0] < roles["center"][0]:
            roles["left"] = excluded_points[0]
        else:
            roles["right"] = excluded_points[0]
    return roles


def classify_4_points(points):
    roles = {}
    best_triplet, excluded_point, best_axis, best_score = find_best_aligned_triplet_from_n(points, 4)
    if best_axis == "x":
        x_locs = best_triplet[:,0]
        ind_low = np.argmin(x_locs)
        ind_hi = np.argmax(x_locs)
        for i in range(3):
            if i == ind_low or i == ind_hi:
                continue
            other_ind = i
        roles["left"] = best_triplet[ind_low]
        roles["right"] = best_triplet[ind_hi]
        roles["center"] = best_triplet[other_ind]
        if excluded_point[0,1] > roles["center"][1]:
            roles["top"] = excluded_point[0]
        else:
            roles["bottom"] = excluded_point[0]
    else:
        y_locs = best_triplet[:,1]
        ind_low = np.argmin(y_locs)
        ind_hi = np.argmax(y_locs)
        for i in range(3):
            if i == ind_low or i == ind_hi:
                continue
            other_ind = i
        roles["bottom"] = best_triplet[ind_low]
        roles["top"] = best_triplet[ind_hi]
        roles["center"] = best_triplet[other_ind]
        if excluded_point[0,0] > roles["center"][0]:
            roles["right"] = excluded_point[0]
        else:
            roles["left"] = excluded_point[0]
    return roles


def classify_5_points(points):
    roles = {}
    best_triplet, excluded_points, best_axis, best_score = find_best_aligned_triplet_from_n(points, 5)
    if best_axis == "x":
        x_locs = best_triplet[:,0]
        ind_low = np.argmin(x_locs)
        ind_hi = np.argmax(x_locs)
        for i in range(3):
            if i == ind_low or i == ind_hi:
                continue
            other_ind = i
        roles["left"] = best_triplet[ind_low]
        roles["right"] = best_triplet[ind_hi]
        roles["center"] = best_triplet[other_ind]
        if excluded_points[0,1] > roles["center"][1]:
            roles["top"] = excluded_points[0]
            roles["bottom"] = excluded_points[1]
        else:
            roles["top"] = excluded_points[1]
            roles["bottom"] = excluded_points[0]
    else:
        y_locs = best_triplet[:,1]
        ind_low = np.argmin(y_locs)
        ind_hi = np.argmax(y_locs)
        for i in range(3):
            if i == ind_low or i == ind_hi:
                continue
            other_ind = i
        roles["bottom"] = best_triplet[ind_low]
        roles["top"] = best_triplet[ind_hi]
        roles["center"] = best_triplet[other_ind]
        if excluded_points[0,0] > roles["center"][0]:
            roles["right"] = excluded_points[0]
            roles["left"] = excluded_points[1]
        else:
            roles["right"] = excluded_points[1]
            roles["left"] = excluded_points[0]
    return roles


def locate_event(frame, params, pyramid):
    pixel_pitch = params["pixel_pitch"]
    sensor_choice = params["sensor_choice"]
    min_sigma = params["min_sigma"]
    lam = params["lam"] # regularization coefficient
    num_steps = params["max_em_steps"]
    num_nbrs = params["num_nbrs"]
    nu = params["nu"]
    coeffs = pyramid.get_coeffs()
    posx_posy, negx_posy, negx_negy, posx_negy = pyramid.posx_posy, pyramid.negx_posy, pyramid.negx_negy, pyramid.posx_negy
    counts = int(frame.sum())
    photon_locs_inds = np.argwhere(frame).astype(float)
    photon_locs_ra = sensor.get_photon_locs(photon_locs_inds, sensor_choice)
    photon_weights = get_photon_weights(photon_locs_ra, num_nbrs, nu)
    all_max_Q_ra = -np.inf*np.ones((3,))
    all_init_loc_ra = np.zeros((3,3))
    all_roles = []
    ii = 0
    cluster_ra = np.arange(3,6)
    for num_clusters in cluster_ra:
        centroids_mm, centroids_pix = get_centroids(photon_locs_ra, photon_weights, sensor_choice, num_clusters)
        if num_clusters == 3:
            roles = classify_3_points(centroids_mm)
        elif num_clusters == 4:
            roles = classify_4_points(centroids_mm)
        elif num_clusters == 5:
            roles = classify_5_points(centroids_mm)
        use_posx = "right" in roles
        use_posy = "top" in roles
        use_negx = "left" in roles
        use_negy = "bottom" in roles

        centroids_mm = []
        for v in roles.values():
            centroids_mm.append(v.reshape(1,2))
        centroids_mm = np.concatenate(centroids_mm, axis=0)
        centroids_pix = np.zeros_like(centroids_mm)
        for i in range(centroids_mm.shape[0]):
            x_coord = centroids_mm[i,0]
            y_coord = centroids_mm[i,1]
            row_idx, col_idx = sensor.convert_coord_to_idx(x_coord, y_coord, sensor_choice)
            centroids_pix[i,0] = row_idx
            centroids_pix[i,1] = col_idx
        init_loc_ra = get_init_locs(roles["center"], pyramid, params["lens_z"], params["S2"])
        Q_init_ra = -np.inf*np.ones((init_loc_ra.shape[0]))
        pred_loc_ra = np.zeros((init_loc_ra.shape[0], 3))
        posx_init = roles["right"] if use_posx else np.zeros((2,))
        posy_init = roles["top"] if use_posy else np.zeros((2,))
        negx_init = roles["left"] if use_negx else np.zeros((2,))
        negy_init = roles["bottom"] if use_negy else np.zeros((2,))
        center_init = roles["center"]
        compute_init_Q = True
        for i in range(init_loc_ra.shape[0]):
            init_loc = init_loc_ra[i]
            locs_steps_ra, Q_ra, all_r_ra = em_algo.run_em(pyramid, photon_locs_ra, photon_weights, coeffs, posx_posy, negx_posy, negx_negy, posx_negy, init_loc, params, sensor_choice, min_sigma, compute_init_Q, use_posx, use_posy, use_negx, use_negy, posx_init, posy_init, negx_init, negy_init, center_init)
            Q = Q_ra[0]
            Q_init_ra[i] = Q
        all_roles.append(roles)
        idx = np.argmax(Q_init_ra)
        init_loc = init_loc_ra[idx]
        all_max_Q_ra[ii] = Q_init_ra[idx]
        all_init_loc_ra[ii] = init_loc
        ii += 1
    idx = np.argmax(all_max_Q_ra)
    final_init_loc = all_init_loc_ra[idx]
    roles = all_roles[idx]
    use_posx = "right" in roles
    use_posy = "top" in roles
    use_negx = "left" in roles
    use_negy = "bottom" in roles
    posx_init = roles["right"] if use_posx else np.zeros((2,))
    posy_init = roles["top"] if use_posy else np.zeros((2,))
    negx_init = roles["left"] if use_negx else np.zeros((2,))
    negy_init = roles["bottom"] if use_negy else np.zeros((2,))
    center_init = roles["center"]
    num_clusters = len(roles)
    compute_init_Q = True
    temp, Q_ra, all_r_ra = em_algo.run_em(pyramid, photon_locs_ra, photon_weights, coeffs, posx_posy, negx_posy, negx_negy, posx_negy, init_loc, params, sensor_choice, min_sigma, compute_init_Q, use_posx, use_posy, use_negx, use_negy, posx_init, posy_init, negx_init, negy_init, center_init)
    Q_init = Q_ra[0]
    compute_init_Q = False
    locs_steps_ra, Q_ra, all_r_ra = em_algo.run_em(pyramid, photon_locs_ra, photon_weights, coeffs, posx_posy, negx_posy, negx_negy, posx_negy, final_init_loc, params, sensor_choice, min_sigma, compute_init_Q, use_posx, use_posy, use_negx, use_negy, posx_init, posy_init, negx_init, negy_init, center_init)
    final_pred_loc = locs_steps_ra[-1]
    Q_ra[0] = Q_init
    r_ra = all_r_ra[-1]
    num_steps_taken = locs_steps_ra.shape[0] - 1
    return final_pred_loc, photon_locs_ra, num_clusters, r_ra, num_steps_taken


def test_captures():
    params, pyramid = init_values()
    sensor_choice = params["sensor_choice"]
    num_nbrs = params["num_nbrs"]
    base_folder = "../images"
    base_save = "../results_data"
    temp_fol_ra = ["pics_thresh_60-80", "pics_thresh_80-100", "pics_thresh_100-120", "pics_thresh_120-140", "pics_thresh_140-160", "pics_thresh_160-180", "pics_thresh_200"]
    comb = np.array(list(combinations(range(4), 2)))
    num_not5 = 0
    num_few_phots = 0
    for f in temp_fol_ra:
        all_photons_counts_ra = []
        event_only_photons_counts_ra = []
        dist_ra = []
        single_frac_removed_ra = []
        double_frac_removed_ra = []
        pred_locs_ra = []
        fol = f"{base_folder}/{f}"
        files = os.listdir(fol)
        print(fol)
        for file in tqdm(files):
            event_all_pred_locs = []
            event_single_frac_removed = []
            event_double_frac_removed = []
            frame = np.load(f"{fol}/{file}")["arr_0"]
            pred_loc, photon_locs_ra, num_clusters, r_ra, num_steps_taken = locate_event(frame, params, pyramid)
            event_all_pred_locs.append(pred_loc.reshape(1,3))
            if num_clusters != 5:
                num_not5 += 1
                # print(f"num not 5: {num_not5}")
                continue
            cluster_assignments = np.argmax(r_ra, axis=1)
            num_all_photons = r_ra.shape[0]
            num_event_only_photons = int((cluster_assignments == 4).sum())
            append_error = True
            # remove one reflection
            for i in range(4):
                inds = cluster_assignments == i
                phots = photon_locs_ra[inds]
                frac_photons_removed = phots.shape[0] / r_ra.shape[0]
                row_ra = []
                col_ra = []
                new_frame = np.copy(frame)
                for q in range(phots.shape[0]):
                    row_idx, col_idx = sensor.convert_coord_to_idx(phots[q,0], phots[q,1], sensor_choice)
                    new_frame[row_idx, col_idx] = 0
                if new_frame.sum() < num_nbrs or not append_error:
                    append_error = False
                    num_few_phots += 1
                    print(f"num_few_photsB: {num_few_phots}")
                    break
                pred_loc_temp, photon_locs_ra_temp, num_clusters_temp, r_ra_temp, num_steps_taken_temp = locate_event(new_frame, params, pyramid)
                event_all_pred_locs.append(pred_loc_temp.reshape(1,3))
                event_single_frac_removed.append(frac_photons_removed)
            # remove two reflections
            for i in range(comb.shape[0]):
                inds0 = cluster_assignments == comb[i,0]
                inds1 = cluster_assignments == comb[i,1]
                phots = photon_locs_ra[np.logical_or(inds0, inds1)]
                frac_photons_removed = phots.shape[0] / r_ra.shape[0]
                row_ra = []
                col_ra = []
                new_frame = np.copy(frame)
                for q in range(phots.shape[0]):
                    row_idx, col_idx = sensor.convert_coord_to_idx(phots[q,0], phots[q,1], sensor_choice)
                    new_frame[row_idx, col_idx] = 0
                if new_frame.sum() < num_nbrs or not append_error:
                    append_error = False
                    num_few_phots += 1
                    print(f"num_few_photsC: {num_few_phots}")
                    break
                pred_loc_temp, photon_locs_ra_temp, num_clusters_temp, r_ra_temp, num_steps_taken_temp = locate_event(new_frame, params, pyramid)
                event_all_pred_locs.append(pred_loc_temp.reshape(1,3))
                event_double_frac_removed.append(frac_photons_removed)
            if append_error:
                event_all_pred_locs = np.concatenate(event_all_pred_locs, axis=0) # pred locs of each individual image of the event
                event_all_mean_loc = np.mean(event_all_pred_locs, axis=0).reshape(1,3) # mean pred loc over each individual image of the event
                event_all_dist_ra = np.linalg.norm(event_all_pred_locs - event_all_mean_loc, axis=1) # distances between each pred loc and mean pred loc
                dist_ra.extend(event_all_dist_ra.tolist())
                single_frac_removed_ra.extend(event_single_frac_removed)
                double_frac_removed_ra.extend(event_double_frac_removed)
                all_photons_counts_ra.append(num_all_photons)
                event_only_photons_counts_ra.append(num_event_only_photons)
                pred_locs_ra.append(pred_loc.reshape(1,3))
        dist_ra = np.array(dist_ra)
        single_frac_removed_ra = np.array(single_frac_removed_ra)
        double_frac_removed_ra = np.array(double_frac_removed_ra)
        all_photons_counts_ra = np.array(all_photons_counts_ra)
        event_only_photons_counts_ra = np.array(event_only_photons_counts_ra)
        pred_locs_ra = np.concatenate(pred_locs_ra, axis=0)
        np.save(f"{base_save}/dist_ra_{f}.npy", dist_ra)
        np.save(f"{base_save}/single_frac_removed_ra_{f}.npy", single_frac_removed_ra)
        np.save(f"{base_save}/double_frac_removed_ra_{f}.npy", double_frac_removed_ra)
        np.save(f"{base_save}/all_photons_counts_ra_{f}.npy", all_photons_counts_ra)
        np.save(f"{base_save}/event_only_photons_counts_ra_{f}.npy", event_only_photons_counts_ra)
        np.save(f"{base_save}/pred_loc_ra_{f}.npy", pred_locs_ra)
        print(f"dist median, mean, std: {np.median(dist_ra)}, {np.mean(dist_ra)}, {np.std(dist_ra)}")
        

def get_results():
    base = "../results_data"
    base_save = "../results_figures"
    thresh_list = ["pics_thresh_60-80", "pics_thresh_80-100", "pics_thresh_100-120", "pics_thresh_120-140", "pics_thresh_140-160", "pics_thresh_160-180", "pics_thresh_200"]
    all_dist_ra = []
    all_single_frac_removed_ra = []
    all_double_frac_removed_ra = []
    for thresh in thresh_list:
        dist_ra = np.load(f"{base}/dist_ra_{thresh}.npy")
        all_dist_ra.append(dist_ra)
        single_frac_remove_ra = np.load(f"{base}/single_frac_removed_ra_{thresh}.npy")
        all_single_frac_removed_ra.append(single_frac_remove_ra)
        double_frac_remove_ra = np.load(f"{base}/double_frac_removed_ra_{thresh}.npy")
        all_double_frac_removed_ra.append(double_frac_remove_ra)

    all_dist_ra = np.concatenate(all_dist_ra)
    all_single_frac_removed_ra = np.concatenate(all_single_frac_removed_ra)
    all_double_frac_removed_ra = np.concatenate(all_double_frac_removed_ra)

    bins_temp = np.arange(0,3.75,.05)
    num_all_dists = len(all_dist_ra)
    plt.figure()
    plt.hist(all_dist_ra, bins=bins_temp, edgecolor='black')
    plt.xlabel("Distance (mm)")
    # plt.savefig(f"{base_save}/all_error_hist_no_title.png", dpi=500, bbox_inches='tight')
    plt.title(f"All Distance \n mean +- std: {np.mean(all_dist_ra):.2f} +- {np.std(all_dist_ra):.2f} mm \n median: {np.median(all_dist_ra):.2f} mm \n {num_all_dists} all combinations")
    plt.savefig(f"{base_save}/all_error_hist.png", dpi=500, bbox_inches='tight')
    plt.close()

    num_single_frac_removed = len(all_single_frac_removed_ra)
    plt.figure()
    plt.hist(all_single_frac_removed_ra, bins=np.arange(0,1,.05), edgecolor='black')
    plt.xlabel("Fraction of counts removed")
    plt.xticks(np.arange(0,1,.1))
    # plt.savefig(f"{base_save}/single_frac_removed_hist_no_title.png", dpi=500, bbox_inches='tight')
    plt.title(f"Single frac removed \n mean +- std: {np.mean(all_single_frac_removed_ra):.2f} +- {np.std(all_single_frac_removed_ra):.2f} \n median: {np.median(all_single_frac_removed_ra):.2f} \n {num_single_frac_removed} single removed combinations")
    plt.savefig(f"{base_save}/single_frac_removed_hist.png", dpi=500, bbox_inches='tight')
    plt.close()

    num_double_frac_removed = len(all_double_frac_removed_ra)
    plt.figure()
    plt.hist(all_double_frac_removed_ra, bins=np.arange(0,1,.05), edgecolor='black')
    plt.xlabel("Fraction of counts removed")
    plt.xticks(np.arange(0,1,.1))
    # plt.savefig(f"{base_save}/double_frac_removed_hist_no_title.png", dpi=500, bbox_inches='tight')
    plt.title(f"Double frac removed \n mean +- std: {np.mean(all_double_frac_removed_ra):.2f} +- {np.std(all_double_frac_removed_ra):.2f} \n median: {np.median(all_double_frac_removed_ra):.2f} \n {num_double_frac_removed} double removed combinations")
    plt.savefig(f"{base_save}/double_frac_removed_hist.png", dpi=500, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    test_captures()
    get_results()


    