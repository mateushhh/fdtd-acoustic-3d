import taichi as ti
from config import N,COURANT_SQ, DT, FREQ_MAX
import math
import numpy as np

ti.init(arch=ti.gpu)

p_old = ti.field(ti.f32, shape=(N, N))
p_curr = ti.field(ti.f32, shape=(N, N))

mask = ti.field(dtype=ti.i32, shape=(N, N)) # gdzie jest sciana

@ti.kernel
def setup_mask():
    for i, j in mask:
        if i <= 1 or i >= N - 2 or j <= 1 or j >= N - 2:
            mask[i, j] = 1
        if N // 4 < i < N // 2 and N // 3 < j < N // 3 + 10:
            mask[i, j] = 1

@ti.kernel
def step(p_prev: ti.template(), p_now: ti.template(), t: ti.f32):
    for i,j in p_now:
        if 0 < i < N - 1 and 0 < j < N - 1:
            if mask[i, j] == 1:   #warunek Dirchleta
                p_prev[i, j] = 0
            else:
                laplacian = p_now[i + 1, j] + p_now[i - 1, j] + \
                            p_now[i, j + 1] + p_now[i, j - 1] - 4 * p_now[i, j]

                p_prev[i, j] = COURANT_SQ * laplacian + 2 * p_now[i, j] - p_prev[i, j]

    #ZRODLO
    center = N // 2
    if t < 0.05:
        p_prev[center, center] += ti.sin(2.0 * math.pi * FREQ_MAX * t) #sygnal sinusoidalny
        # jest soft source, += sprawia ze fala, która wróci, przejdzie po prostu przez głosnik



def main():
    gui = ti.GUI("2d - fdtd", res=(N, N))
    setup_mask()
    buffers = [p_old, p_curr]
    print(f"Rozmiar siatki: {N}x{N}")

    frame = 0
    while gui.running:
        b_old = frame % 2
        b_curr = (frame + 1) % 2

        step(buffers[b_old], buffers[b_curr], frame * DT)

        p_np = buffers[b_old].to_numpy()
        max_val = np.abs(p_np).max()
        p_normalized = p_np / (max_val + 1e-8)
        p_np_norm = p_normalized * 0.5 + 0.5

        mask_np = mask.to_numpy() # zeby bylo widac sciany

        img = np.where(mask_np > 0, 1.0, p_np_norm)
        gui.set_image(img)

        gui.show()
        frame += 1

if __name__ == "__main__":
    main()