import numpy as np
import matplotlib.pyplot as plt
import raytracing_cython as rt
import time


if __name__ == '__main__':

    file_name = "timing_cython.txt"
    iters = 10
    with open(file_name, "w+") as f:
                    f.truncate(0)  # Clear the contents of the file
                    for i in range(iters):
                        ts = time.perf_counter()
                        img = rt.run()
                        te = time.perf_counter()
                        f.write('%.6f' % (te-ts))
                        f.write('\n')

    # img = rt.run()
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.imshow(img)
    # ax.set_axis_off()
    # plt.show()