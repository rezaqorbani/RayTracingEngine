# cython: language_level=3
import numpy as np
import cython

cdef double[::1] position = np.array([0., 0., 1.])
cdef double radius = 1.
cdef double[::1] color = np.array([0., 0., 1.])
cdef double diffuse = 1.
cdef double specular_c = 1.
cdef int specular_k = 50
# Sphere properties.
    
# Light position and color.
cdef double[::1] L = np.array([5., 5., -10.])
cdef double[::1] color_light = np.array([1., 1., 1.])
cdef double ambient = .05
    
# Camera.
cdef double[::1] O = np.array([0., 0., -1.])  # Position.
cdef double[::1] Q = np.array([0., 0., 0.])  # Pointing to.

# Size of the screen in pixels.
cdef int w = 400
cdef int h = 400 

cdef double[::1] clip(double[::1] a, int min_value, int max_value):
    cdef int j
    cdef double[::1] retval = np.empty(a.shape[0])

    for j in range(a.shape[0]):
        retval[j] = min(max(a[j], min_value), max_value)
    
    return retval


cdef double[::1] add(double[::1] a, double[::1] b):
    cdef int j
    cdef double[::1] retval = np.empty(a.shape[0])

    for j in range(a.shape[0]):
        retval[j] = a[j] + b[j]

    return retval

cdef double[::1] substract(double[::1] a, double[::1] b):
    cdef int j
    cdef double[::1] retval = np.empty(a.shape[0])

    for j in range(a.shape[0]):
        retval[j] = a[j] - b[j]

    return retval


cdef double[::1] multiply(double[::1] a, double[::1] b):
    cdef Py_ssize_t j
    cdef double[::1] retval = np.empty(a.shape[0])

    for j in range(a.shape[0]):
        retval[j] = a[j] * b[j]

    return retval
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def normalize( double[::1] x):
        # This function normalizes a vector.
        x /= np.linalg.norm(x)
        return x
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def intersect_sphere(  double[::1] O,  double[::1] D,
                       double[::1] S,   double R):
        # Return the distance from O to the intersection
        # of the ray (O, D) with the sphere (S, R), or
        # +inf if there is no intersection.
        # O and S are 3D points, D (direction) is a
        # normalized vector, R is a scalar.
        cdef double[::1] OS 
        cdef double a
        cdef double b
        cdef double c
        cdef double disc
        cdef double distSqrt
        cdef double q
        cdef double t0
        cdef double t1


        a = np.dot(D, D)
        OS = substract(O , S)
        b = 2 * np.dot(D, OS)
        c = np.dot(OS, OS) - R*R
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def trace_ray(  double[::1] O,  double[::1] D):
        cdef double[::1] col
        cdef double t
        cdef double[::1] M
        cdef double[::1] N
        cdef double[::1] toL
        cdef double[::1] toO
        cdef double[::1] term
        
        # Find first point of intersection with the scene.
        t = intersect_sphere(O, D, position, radius)
        # No intersection?
        if t == np.inf:
            return
        # Find the point of intersection on the object.

        M = add(O, D * np.full_like(D, t))
        N = normalize(substract(M, position))
        toL = normalize(substract(L,M))
        toO = normalize(substract(O,M))
        term = np.full_like(color, diffuse) * max(np.dot(N, toL), 0) * color
        # Ambient light.
        col = np.full_like(color, ambient)
        # Lambert shading (diffuse).
        col = add(col, term )
        # Blinn-Phong shading (specular).
        col = add(col, multiply( multiply(np.full_like(color_light, specular_c) , color_light) , \
            np.full_like( color_light, max(np.dot(N, normalize(add(toL , toO))), 0) \
            ** specular_k)))
        return col

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def run():
        cdef double[:, :, :] img = np.empty((h, w, 3))
        # Loop through all pixels.
        cdef int i, j
        cdef double x, y
        cdef double[::1] col
        cdef double[::1] D

        for i, x in enumerate(np.linspace(-1, 1, w)):
            for j, y in enumerate(np.linspace(-1, 1, h)):
                # Position of the pixel.
                Q[0], Q[1] = x, y
                # Direction of the ray going through
                # the optical center.
                D = normalize(substract(Q, O))
                # Launch the ray and get the color
                # of the pixel.
                col = trace_ray(O, D)
                if col is None:
                    continue
                img[h - j - 1, i, :] = clip(col, 0, 1)
        return img