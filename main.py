import taichi as ti
from visualization import Simulation, CubeGeometry, SceneRenderer,config


def main():
    ti.init(arch=ti.gpu, kernel_profiler=True, device_memory_GB=config.MEMORY_LIMIT_GB)

    sim = Simulation()
    geo = CubeGeometry()
    renderer = SceneRenderer()

    # Debug
    frame_id = 0 #


    sim.init_room()
    while renderer.is_running:
        sim.update()

        renderer.render_frame(simulation=sim, geometry=geo)

        # Debug
        frame_id += 1
        if frame_id % 100 == 0:
            ti.profiler.print_kernel_profiler_info()
            ti.profiler.clear_kernel_profiler_info()


if __name__ == "__main__":
    main()