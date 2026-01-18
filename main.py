import taichi as ti
from visualization import Simulation, CubeGeometry, SceneRenderer,config


def main():
    ti.init(arch=ti.gpu, device_memory_GB=config.MEMORY_LIMIT_GB)

    sim = Simulation()
    geo = CubeGeometry()
    renderer = SceneRenderer()

    while renderer.is_running:
        sim.update()

        renderer.render_frame(simulation=sim, geometry=geo)


if __name__ == "__main__":
    main()