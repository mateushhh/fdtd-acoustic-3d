import taichi as ti
from . import config

@ti.data_oriented
class Simulation:
    def __init__(self):
        #Deklaracja pamieci poczatkowej
        self.pos = ti.Matrix.field(4, 4, dtype=ti.f32, shape=config.MAX_VOXELS)
        #Counter w pamieci ktory jest atomowy(nie wysypuje sie przy watkach)
        self.count = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def update(self):
        """
        Ta petla wykonuje sie 60 razy na sekunde - po prostu update.
        Raczej bedziemy w tym miejscu dodawac tworzenie fal itd.
        :return:
        """
        self.count[None] = 0
        for i, j, k in ti.ndrange(config.N, config.N, config.N):
            #filtering voxels - example

            """
            w tym miejscu mozecie sobie filtrowac te voxele ktore maja sie pojawiac.
            Mozecie sobie poeksperymentowac z wartosciami. Aktualnie pokazuje zarys szescianu.
            Jesli dacie can_draw=True to macie pelny
            """
            cond_x = (i == 0 or i == config.N - 1)
            cond_y = (j == 0 or j == config.N - 1)
            cond_z = (k == 0 or k == config.N - 1)

            can_draw = (cond_x and cond_y) or (cond_x and cond_z) or (cond_y and cond_z)
            # can_draw = True
            if can_draw:
                index = ti.atomic_add(self.count[None], 1)
                x = i * config.VOXEL_DISTANCE
                y = j * config.VOXEL_DISTANCE
                z = k * config.VOXEL_DISTANCE
                self.pos[index] = ti.Matrix([
                    [1.0, 0.0, 0.0, x],
                    [0.0, 1.0, 0.0, y],
                    [0.0, 0.0, 1.0, z],
                    [0.0, 0.0, 0.0, 1.0]
                ])