import taichi as ti
import math

def get_time_step(dimensions, dx, speed, safety_factor):
    courant_limit = 1.0 / math.sqrt(dimensions)
    return (dx / speed) * courant_limit * safety_factor

C = 343.0   # predkosc [m/s]
FREQ_MAX = 1000.0 # maksymalna czestotliwpsc fali [Hz]
SIZE_M = 20.0  # rozmiar [m]
SIGMA = math.sqrt(2 * math.log(2)) / (2 * math.pi * FREQ_MAX)
AMPLITUDE = 10.0

NODES_PER_WAVELENGTH = 10    # ilsoc punktow na dlugosc fali
SAFETY_FACTOR = 0.99 # wspolczynnik bezpieczenstwa
DIM = 2  # 2D

WAVELENGTH = C / FREQ_MAX    # dlugosc fali [m]
DX = WAVELENGTH / NODES_PER_WAVELENGTH   # krok odleglosciowy [m]

N = int(SIZE_M / DX)     # ilosc punktow w jednym wymiarze
N_2 = N + 2 # dodajemy po jednym punckie na kazda strone do obliczen

DT = get_time_step(DIM, DX, C, SAFETY_FACTOR)

COURANT = (1.0 / math.sqrt(DIM)) * SAFETY_FACTOR
COURANT_SQ = COURANT**2

#material( 500 HZ) https://www.acoustic-supplies.com/absorption-coefficient-chart/
brick_alpha = 0.03 # brick(natural)
wooden_bench_alpha = 0.66 # Benches (wooden, fully occupied)

SRC_X = N_2 // 2
SRC_Y = N_2 // 2

frame_duration = 0.02 # w sekundach
delay_gauss = 4 * SIGMA

def print_config():
    print("-" * 30)
    print(f"SYMULACJA FDTD {DIM}D")
    print("-" * 30)
    print(f"Rozmiar siatki:  {N} x {N}")
    print(f"Krok DX:         {DX*1000:.2f} mm")
    print(f"Krok DT:         {DT*1e6:.2f} us")
    print(f"Liczba Couranta: {COURANT:.4f} (Max: {1/math.sqrt(DIM):.4f})")
    print("-" * 30)

if __name__ == "__main__":
    print_config()