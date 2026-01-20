import taichi as ti
from . import config

@ti.data_oriented
class Simulation:
    def __init__(self):
        # Deklaracja pamieci poczatkowej
        self.voxel_pos = ti.Matrix.field(4, 4, dtype=ti.f32, shape=config.MAX_VOXELS)
        # Counter w pamieci ktory jest atomowy (nie wysypuje sie przy watkach) potrzebny do otoczenia voxele
        self.voxel_count = ti.field(dtype=ti.i32, shape=())

        self.particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=config.MAX_VOXELS)
        self.particle_colors = ti.Vector.field(3, dtype=ti.f32, shape=config.MAX_VOXELS)
        self.particle_count = ti.field(dtype=ti.i32, shape=())

        # Czas symulacji
        self.t = ti.field(float, shape=())

        # Ciśnienie w punkcie w przestrzeni
        self.pressure = ti.field(dtype=ti.f32, shape=config.MAX_VOXELS)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=config.MAX_VOXELS)


    @ti.kernel
    def update(self):
        """
        Ta petla wykonuje sie 60 razy na sekunde - po prostu update.
        Raczej bedziemy w tym miejscu dodawac tworzenie fal itd.
        :return:
        """

        self.voxel_count[None] = 0
        self.particle_count[None] = 0

        self.t[None] += config.DELTA_TIME
        current_t = self.t[None]

        source = ti.Vector([config.SOURCE_POS[0], config.SOURCE_POS[1], config.SOURCE_POS[2]])

        for i, j, k in ti.ndrange(config.N, config.N, config.N):

            x = i * config.VOXEL_DISTANCE
            y = j * config.VOXEL_DISTANCE
            z = k * config.VOXEL_DISTANCE

            """
            Maciej: w tym miejscu mozecie sobie filtrowac te voxele ktore maja sie pojawiac.
            Mozecie sobie poeksperymentowac z wartosciami.
            Jesli dacie can_draw=True to macie pelny
            
            Mati: zmieniłem can_draw na is_voxel żeby generować woksele osobno i particle osobno
            nie generujemy powietrza w tych samych miejscach co są ściany
            """

            is_voxel = (j == 0)

            # Tymczasowo podłoga (y==0) jest voxelami a reszta jest particlesami "powietrzem"
            if is_voxel:
                index = ti.atomic_add(self.voxel_count[None], 1)

                self.voxel_pos[index] = ti.Matrix([
                    [1.0, 0.0, 0.0, x],
                    [0.0, 1.0, 0.0, y],
                    [0.0, 0.0, 1.0, z],
                    [0.0, 0.0, 0.0, 1.0]
                ])
            else:
                index = ti.atomic_add(self.particle_count[None], 1)

                # Generowanie próbnej fali (dla przykładu jest statyczna)
                current_pos = ti.Vector([x,y,z])
                distance_from_source = (current_pos - source).norm()

                # Wartość wygenerowanej fali w punkcie (obliczona strata na podstawie przesunięcia w czasie i dystansie)
                # skalary tutaj to testowe wartości
                wave_val = ti.sin(distance_from_source)
                pressure = wave_val / (distance_from_source * 0.1 + 1.0)

                if y == config.N / 2: # Tymczasowa wizualizacja na płaszczyźnie y
                    if ti.abs(pressure) > config.PRESSURE_TRESHOLD:
                        r, g, b = 0.0, 0.0, 0.0
                        if pressure > 0.0:
                            r = ti.abs(pressure)
                        else:
                            b = ti.abs(pressure)

                        self.particle_pos[index] = ti.Vector([x, y ,z])
                        self.particle_colors[index] = ti.Vector([r, g, b])