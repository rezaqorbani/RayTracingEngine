import numpy as np
import numba
import matplotlib.pyplot as plt
import time
import os

w, h = 400, 400  # Size of the screen in pixels.

# a timing decorator using time.perf_counter()
def timer(file_name, iter):
    def timeit_decorator(method):
        def timed(*args, **kw):
            with open(file_name, "r+") as f:
                f.truncate(0)  # Clear the contents of the file
                for i in range(iter):
                    ts = time.perf_counter()
                    result = method(*args, **kw)
                    te = time.perf_counter()
                    f.write(te-ts)
                    f.write('\n')
        return timed
    return timeit_decorator


@numba.jit(nopython=True)
@profile
def normalize(x):
        # This function normalizes a vector.
        x /= np.linalg.norm(x)
        return x

@numba.jit(nopython=True)
@profile
def clip(x, x_min, x_max):
        return np.minimum(x_max, np.maximum(x, x_min))

@numba.jit(nopython=True)
@profile
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

@numba.jit(nopython=True)
@profile
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

        term = diffuse * max(np.dot(N, toL), 0) * color

        # Ambient light.
        col = np.full_like(color, ambient)
        # Lambert shading (diffuse).
        col += term
        # Blinn-Phong shading (specular).
        col += specular_c * color_light * \
            max(np.dot(N, normalize(toL + toO)), 0) \
            ** specular_k
        return col

@numba.jit(nopython=True)
@profile
def run(O, Q):
        img = np.zeros((h, w, 3))
        # Loop through all pixels.
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
                img[h - j - 1, i, :] = clip(col, 0, 1)
        return img

if __name__ == '__main__':
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
    O = np.array([0., 0., -1.] )  # Position.
    Q = np.array([0., 0., 0.] ) 

    # run the code for the first time to compile the code 
    img = run(O, Q)

    # time the compiled code
    file_name = "timing_numba.txt"
    iters = 10
    with open(file_name, "w+") as f:
                    f.truncate(0)  # Clear the contents of the file
                    for i in range(iters):
                        ts = time.perf_counter()
                        img = run(O, Q)
                        te = time.perf_counter()
                        f.write('%.6f' % (te-ts))
                        f.write('\n')
    


