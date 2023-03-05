import numpy as np
import cython

cdef int w = 400
cdef int h = 400
# Sphere properties.
position = np.array([0., 0., 1.])
radius = 1.
color = np.array([0., 0., 1.])
diffuse = 1.
specular_c = 1.
specular_k = 50
    
# Light position and color.
L = np.array([5., 5., -10.])
color_light = np.ones(3)
ambient = .05
    
# Camera.
O = np.array([0., 0., -1.])  # Position.
Q = np.array([0., 0., 0.])  # Pointing to.

def normalize(x):
        # This function normalizes a vector.
        x /= np.linalg.norm(x)
        return x

def intersect_sphere(O, D, S, R):
        # Return the distance from O to the intersection
        # of the ray (O, D) with the sphere (S, R), or
        # +inf if there is no intersection.
        # O and S are 3D points, D (direction) is a
        # normalized vector, R is a scalar.
        a = np.dot(D, D)
        OS = O - S
        b = 2 * np.dot(D, OS)
        c = np.dot(OS, OS) - R * R
        disc = b * b - 4 * a * c
        if disc > 0:
            distSqrt = np.sqrt(disc)
            q = (-b - distSqrt) / 2.0 if b < 0 \
                else (-b + distSqrt) / 2.0
            t0 = q / a
            t1 = c / q
            t0, t1 = min(t0, t1), max(t0, t1)
            if t1 >= 0:
                return t1 if t0 < 0 else t0
        return np.inf

def trace_ray(O, D):
        # Find first point of intersection with the scene.
        t = intersect_sphere(O, D, position, radius)
        # No intersection?
        if t == np.inf:
            return
        # Find the point of intersection on the object.
        M = O + D * t
        N = normalize(M - position)
        toL = normalize(L - M)
        toO = normalize(O - M)
        # Ambient light.
        col = ambient
        # Lambert shading (diffuse).
        col += diffuse * max(np.dot(N, toL), 0) * color
        # Blinn-Phong shading (specular).
        term = specular_c * color_light 
        term = term * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k
        col += term
        return col

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def run():
        # Loop through all pixels.
        cdef double[:, :, :] img = np.empty((h, w, 3))
        cdef int i, j
        cdef double x, y
        cdef double[:] arr
        
        for i, x in enumerate(np.linspace(-1, 1, w)):
            for j, y in enumerate(np.linspace(-1, 1, h)):
                # Position of the pixel.
                Q[0], Q[1] = x, y
                # Direction of the ray going through
                # the optical center.
                D = normalize(Q - O)
                # Launch the ray and get the color
                # of the pixel.
                col = trace_ray(O, D)
                if col is None:
                    continue
                arr = np.clip(col, 0, 1)
                img[h - j - 1, i, : ] = arr
        return img
