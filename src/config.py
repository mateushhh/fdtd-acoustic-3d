import taichi as ti
import math

def get_time_step(dimensions, dx, speed, safety_factor):
    courant_limit = 1.0 / math.sqrt(dimensions)
    return (dx / speed) * courant_limit * safety_factor

C = 343.0   # predkosc [m/s]
FREQ_MAX = 1000.0 # maksymalna czestotliwpsc fali [Hz]
SIZE_M = 20.0  # rozmiar [m]

NODES_PER_WAVELENGTH = 10    # ilsoc punktow na dlugosc fali
SAFETY_FACTOR = 0.99 # wspolczynnik bezpieczenstwa
DIM = 2  # 2D

WAVELENGTH = C / FREQ_MAX    # dlugosc fali [m]
DX = WAVELENGTH / NODES_PER_WAVELENGTH   # krok odleglosciowy [m]

N = int(SIZE_M / DX)     # ilosc punktow w jednym wymiarze

DT = get_time_step(DIM, DX, C, SAFETY_FACTOR)

COURANT = (1.0 / math.sqrt(DIM)) * SAFETY_FACTOR
COURANT_SQ = COURANT**2

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