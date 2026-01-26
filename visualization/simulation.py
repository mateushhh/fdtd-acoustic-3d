import taichi as ti
from . import config

@ti.data_oriented
class Simulation:
    def __init__(self):
        """
        indeksowanie: [i,j,k]    ( 0 -> air, 1 -> solid )
        """
        self.is_voxel = ti.field(dtype=ti.i32, shape=(config.N, config.N, config.N))

        self.voxel_pos = ti.Matrix.field(4, 4, dtype=ti.f32, shape=config.MAX_VOXELS)
        self.voxel_count = ti.field(dtype=ti.i32, shape=())

        """
        indeksowanie: [p_idx] # idx = i * config.N * config.N + j * config.N + k
        """
        self.particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=config.MAX_VOXELS)
        self.particle_pos_home = ti.Vector.field(3, dtype=ti.f32, shape=config.MAX_VOXELS)
        self.particle_colors = ti.Vector.field(3, dtype=ti.f32, shape=config.MAX_VOXELS)

        self.t = ti.field(float, shape=())

    @ti.kernel
    def init_room(self):
        """
        Ta funkcja generuje wszystkie cząsteczki oraz woksele w miejscach wcześniej zdefiniowanych
        Cząsteczek jest N^3 znajdują się nawet tam gdzie ściany, ale są przesunięte na -1000,-1000,-1000
        Cząsteczki powietrza których nie chcemy widzieć na symulacji też są przesunięte na -1000,-1000,-1000
        Wszystkie woksele i cząsteczki są wczytane do pamięci na starcie i jedyne co jest w nich modyfikowane
        to pozycja cząsteczki i jej kolor w funkcji update()
        """

        self.voxel_count[None] = 0

        for i, j, k in ti.ndrange(config.N, config.N, config.N):
            p_idx = i * config.N * config.N + j * config.N + k

            x = i * config.VOXEL_DISTANCE
            y = j * config.VOXEL_DISTANCE
            z = k * config.VOXEL_DISTANCE

            self.is_voxel[i, j, k] = (j == 0)

            if self.is_voxel[i, j, k] == 1:
                v_idx = ti.atomic_add(self.voxel_count[None], 1)
                self.voxel_pos[v_idx] = ti.Matrix([
                    [1.0, 0.0, 0.0, x],
                    [0.0, 1.0, 0.0, y],
                    [0.0, 0.0, 1.0, z],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                self.particle_pos[p_idx] = ti.Vector([-1000.0, -1000.0, -1000.0])
            else:
                self.particle_pos_home[p_idx] = ti.Vector([x, y, z])
                self.particle_pos[p_idx] = ti.Vector([-1000.0, -1000.0, -1000.0])


    @ti.kernel
    def update(self):
        """
        Ta petla wykonuje sie 60 razy na sekunde - po prostu update.
        Raczej bedziemy w tym miejscu dodawac tworzenie fal itd.
        """

        self.t[None] += config.DELTA_TIME
        current_t = self.t[None]

        source = ti.Vector([config.SOURCE_POS[0], config.SOURCE_POS[1], config.SOURCE_POS[2]])

        for i, j, k in ti.ndrange(config.N, config.N, config.N):
            p_idx = i * config.N * config.N + j * config.N + k

            if self.is_voxel[i, j, k] == 0:
                home = self.particle_pos_home[p_idx]

                """
                Generowanie próbnej "fali" sinus
                Wartość wygenerowanej fali w punkcie (obliczona strata na podstawie przesunięcia w czasie i dystansie)
                """
                distance_from_source = (home - source).norm()
                wave_val = ti.sin(current_t - distance_from_source * 0.5)
                pressure = wave_val / (distance_from_source * 0.1 + 1.0)

                """
                Tymczasowa wizualizacja na płaszczyźnie y
                Sprawdzenie czy wartość ciśnienia jest powyżej progu bramki szumów "config.PRESSURE_THRESHOLD"
                """
                is_visible = False
                if j <= config.N // 2:
                    if ti.abs(pressure) >= config.PRESSURE_THRESHOLD:
                        is_visible = True

                if is_visible:
                    self.particle_pos[p_idx] = home
                    r, g, b = 0.0, 0.0, 0.0
                    if pressure > 0.0:
                        r = ti.abs(pressure)
                    else:
                        b = ti.abs(pressure)

                    self.particle_pos[p_idx] = self.particle_pos_home[p_idx]
                    self.particle_colors[p_idx] = ti.Vector([r, g, b])
                else:
                    self.particle_pos[p_idx] = ti.Vector([-1000.0, -1000.0, -1000.0])