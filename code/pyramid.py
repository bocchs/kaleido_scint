import numpy as np

# equilateral pyramid vertex at origin, base side at +z
# base is square
class Pyramid():
    def __init__(self, height, base_len, flat_vertex_z, refrac_index):
        self.n = refrac_index # index of refraction
        orig = np.array([0, 0, 0])
        self.top_z = orig[2] + height
        self.base_len = base_len # square length
        c = base_len / 2
        h = height
        # corners of pyramid base (pyramid opening)
        self.posx_posy = np.array([c, c, h])
        self.negx_posy = np.array([-c, c, h])
        self.negx_negy = np.array([-c, -c, h])
        self.posx_negy = np.array([c, -c, h])
        # extended corners of flat vertex base (triangle)
        v1 = np.array([-base_len, 0, flat_vertex_z], dtype=float)
        v2 = np.array([base_len, base_len, flat_vertex_z], dtype=float)
        v3 = np.array([base_len, -base_len, flat_vertex_z], dtype=float)
        # looking down into pyramid from +z, define faces counter clockwise
        self.posx_face = np.vstack((orig, self.posx_negy, self.posx_posy)) # face on +x side
        self.posx_norm = np.cross(self.posx_negy-orig, self.posx_posy-orig) # norm facing inward into pyramid
        self.posx_norm /= np.linalg.norm(self.posx_norm)
        self.posy_face = np.vstack((orig, self.posx_posy, self.negx_posy))
        self.posy_norm = np.cross(self.posx_posy-orig, self.negx_posy-orig)
        self.posy_norm /= np.linalg.norm(self.posy_norm)
        self.negx_face = np.vstack((orig, self.negx_posy, self.negx_negy))
        self.negx_norm = np.cross(self.negx_posy-orig, self.negx_negy-orig)
        self.negx_norm /= np.linalg.norm(self.negx_norm)
        self.negy_face = np.vstack((orig, self.negx_negy, self.posx_negy))
        self.negy_norm = np.cross(self.negx_negy-orig, self.posx_negy-orig)
        self.negy_norm /= np.linalg.norm(self.negy_norm)
        self.flat_vertex_face = np.vstack((v1, v2, v3))
        self.flat_vertex_norm = np.array([0, 0, 1], dtype=float)
        self.faces = [self.posx_face, self.posy_face, self.negx_face, self.negy_face]
        self.face_norms = [self.posx_norm, self.posy_norm, self.negx_norm, self.negy_norm]
        self.face_strings = ["posx", "posy", "negx", "negy"]
        self.verts = np.vstack((orig, self.posx_posy, self.negx_posy, self.negx_negy, self.posx_negy))

    # get x,y,z coeffs for transforming a location over a mirror
    def get_coeffs(self):
        x_posx_coeffs, y_posx_coeffs, z_posx_coeffs = get_mirror_functions(self.posx_norm)
        x_posy_coeffs, y_posy_coeffs, z_posy_coeffs = get_mirror_functions(self.posy_norm)
        x_negx_coeffs, y_negx_coeffs, z_negx_coeffs = get_mirror_functions(self.negx_norm)
        x_negy_coeffs, y_negy_coeffs, z_negy_coeffs = get_mirror_functions(self.negy_norm)
        coeffs = np.concatenate([x_posx_coeffs.T, y_posx_coeffs.T, z_posx_coeffs.T, \
                  x_posy_coeffs.T, y_posy_coeffs.T, z_posy_coeffs.T, \
                  x_negx_coeffs.T, y_negx_coeffs.T, z_negx_coeffs.T, \
                  x_negy_coeffs.T, y_negy_coeffs.T, z_negy_coeffs.T], axis=0)
        return coeffs

    def point_in_pyramid(self, p):
        if p[2] > self.top_z or p[2] < 0:
            return False
        for i in range(len(self.faces)):
            point_on_face = self.faces[i][0]
            face_n = self.face_norms[i] # face normal vector pointing inward
            dir_vec = p - point_on_face # vector pointing from face to point
            if np.dot(dir_vec, face_n) <= 0:
                return False
        return True


# returns true if p lies inside pyramid, false otherwise
def point_in_pyramid(p, pyramid):
    if p[2] > pyramid.top_z or p[2] < 0:
        return False
    for i in range(len(pyramid.faces)):
        point_on_face = pyramid.faces[i][0]
        face_n = pyramid.face_norms[i] # face normal vector pointing inward
        dir_vec = p - point_on_face # vector pointing from face to point
        if np.dot(dir_vec, face_n) <= 0:
            return False
    return True


# origin at bottom of scintillator
def get_mirror_functions(pyramid_face_norm):
    transform = np.zeros((4,4))
    # reflection matrix
    transform[:3,:3] = np.eye(3) - 2 * pyramid_face_norm.reshape(-1,1) @ pyramid_face_norm.reshape(-1,1).T
    x_out_coeffs = transform[0,:3].reshape(-1,1) # coeffs w1, w2, w3 corresponding to x = w1*x + w2*y + w3*z
    y_out_coeffs = transform[1,:3].reshape(-1,1) # coeffs w1, w2, w3 corresponding to y = w1*x + w2*y + w3*z
    z_out_coeffs = transform[2,:3].reshape(-1,1) # coeffs w1, w2, w3 corresponding to z = w1*x + w2*y + w3*z
    return x_out_coeffs, y_out_coeffs, z_out_coeffs

