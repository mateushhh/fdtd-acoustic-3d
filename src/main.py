import taichi as ti
from config import N,COURANT_SQ, DT, FREQ_MAX, N_2, brick_alpha, wooden_bench_alpha, COURANT, SIGMA, AMPLITUDE, SRC_X, SRC_Y, DT, frame_duration, delay_gauss
import math
import numpy as np
import time

ti.init(arch=ti.gpu)

p_old = ti.field(ti.f32, shape=(N_2, N_2))
p_curr = ti.field(ti.f32, shape=(N_2, N_2))

k_field = ti.field(dtype=ti.i32, shape=(N_2, N_2)) #ilosc sasiadow
alpha_field = ti.field(dtype=ti.f32, shape=(N_2, N_2)) # współczynniki pochlaniania
bk_field = ti.field(dtype=ti.f32, shape=(N_2, N_2)) # macierz bk

@ti.kernel
def generation_symulation_map():
    for i,j in k_field: # domyslnie wszedzie 4 sąsiadów i otwarta przestrzen
        k_field[i , j] = 4
        alpha_field [i, j] = 1.0

    for i,j in k_field: # warstwa zewnetrzna potrzebna do obliczeń
        if i == 0 or j == 0 or i == N_2 - 1 or j == N_2 - 1:
            k_field[i, j] = 0
            alpha_field[i, j] = 0.0

    for i,j in k_field: # sciany
        if((i==1 and j ==1) or (i==1 and j==N_2-2) or (i==N_2-2 and j==1) or (i==N_2-2 and j==N_2-2)):
            k_field[i, j] = 2
            alpha_field[i, j] = brick_alpha

        elif (i == 1 or i == N_2 - 2) and (j > 0 and j < N_2 -1):
            k_field[i, j] = 3
            alpha_field[i, j] = brick_alpha

        elif (j == 1 or j == N_2 - 2) and (i > 0 and i < N_2 -1):
            k_field[i, j] = 3
            alpha_field[i, j] = brick_alpha

    r_x1, r_x2 = 100, 200
    r_y1, r_y2 = 100, 200

    for i,j in k_field:
        if i > r_x1 and i < r_x2 and j > r_y1 and j < r_y2:
            k_field[i, j] = 0
            alpha_field[i, j] = 0.0

        elif (i == r_x1 and j == r_y1) or \
                (i == r_x1 and j == r_y2) or \
                (i == r_x2 and j == r_y1) or \
                (i == r_x2 and j == r_y2):
            k_field[i, j] = 4
            alpha_field[i, j] = wooden_bench_alpha

        elif (i == r_x1 or i == r_x2) and (j > r_y1 and j < r_y2):
            k_field[i, j] = 3
            alpha_field[i, j] = wooden_bench_alpha

        elif (j == r_y1 or j == r_y2) and (i > r_x1 and i < r_x2):
            k_field[i, j] = 3
            alpha_field[i, j] = wooden_bench_alpha




    for i,j in bk_field:
        k_val = k_field[i, j]
        beta_val = beta_from_alpha(alpha_field[i, j])

        bk_field[i, j] = (4.0 - float(k_val)) * COURANT * beta_val / 2.0

@ti.func
def beta_from_alpha(alpha):
    Beta = 0.0
    if alpha >= 1.0:  # otwarta przestrzeń
        Beta = 1.0
    elif alpha <= 0.0:    # idealna ściana
        Beta = 0.0
    else:
        R = ti.sqrt(1.0 - alpha)
        Beta = (1.0 - R) / (1.0 + R)

    return Beta



@ti.kernel
def step(p_prev: ti.template(), p_now: ti.template(), steps: int):
    for i,j in p_now:
        if k_field[i , j] > 0:
            k_val = k_field[i,j]
            bk_val = bk_field[i,j]
            w1 =(2.0 - float(k_val) * COURANT_SQ) * p_now[i, j]
            w2 = (bk_val - 1.0) * p_prev[i, j]
            w3 = COURANT_SQ * (p_now[i + 1, j] + p_now[i - 1, j] + p_now[i, j + 1] + p_now[i, j - 1])
            p_prev[i, j] = (w1 + w2 + w3) / (1 + bk_val)
        else:
            p_prev[i, j] = 0

    #ZRODLO
    pulse = AMPLITUDE * ti.exp(- ((DT*steps - DT*delay_gauss)**2) / (2 * SIGMA**2))
    p_prev[SRC_X, SRC_Y] += pulse # soft source


def main():
    gui = ti.GUI("2d - fdtd", res=(N_2, N_2))
    generation_symulation_map()
    buffers = [p_old, p_curr]
    print(f"Rozmiar siatki: {N}x{N}")

    prev_time = time.time()

    steps = 0
    accumulator = 0

    alpha_map = alpha_field.to_numpy()
    is_wall = alpha_map < 0.99

    while gui.running:

        now = time.time()
        delta_real_time = now - prev_time
        prev_time = now

        accumulator+=delta_real_time
        if(accumulator >= frame_duration):
            steps += 1
            b_old = steps % 2
            b_curr = (steps + 1) % 2

            step(buffers[b_old], buffers[b_curr], steps)

            p_np = buffers[b_old].to_numpy()
            max_val = np.abs(p_np).max()
            p_normalized = p_np / (max_val + 1e-8)

            gray_val = p_normalized * 0.5 + 0.5

            gray_val = np.clip(gray_val, 0.0, 1.0) # zabezpieczenie choc i tak juz wczesniej znormalizowane


            img = np.dstack((gray_val, gray_val, gray_val))

            img[is_wall, 1] *= 0.3
            img[is_wall, 2] *= 0.2

            gui.set_image(img)
            gui.show()


            accumulator = 0





if __name__ == "__main__":
    main()