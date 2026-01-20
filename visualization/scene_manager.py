import taichi as ti
from taichi.examples.ggui_examples.mass_spring_game_ggui import per_vertex_color

from . import config,geometry,simulation


class SceneRenderer:
    def __init__(self):
        self.window = ti.ui.Window(
            name="demo",
            res=config.SCREEN_RESOLUTION,
            fps_limit=config.FPS_LIMIT
        )
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()

        self.camera.position(config.CAMERA_POS_X, config.CAMERA_POS_Y, config.CAMERA_POS_Z)
        self.camera.lookat(config.CAMERA_LOOKAT_X, config.CAMERA_LOOKAT_Y, config.CAMERA_LOOKAT_Z)

    def render_frame(self, simulation, geometry):
        """
        Render the current frame
        :return:
        """
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        self.scene.point_light(pos=(2, 5, 2), color=config.LIGHT_COLOR)
        self.scene.ambient_light(config.AMBIENT_COLOR)

        # Rysowanie voxeli
        self.scene.mesh_instance(
            vertices=geometry.verts,
            indices=geometry.indices,
            transforms=simulation.voxel_pos,
            instance_count=simulation.voxel_count[None],
            color=config.CUBE_COLOR
        )
        # Rysowanie particlesów
        self.scene.particles(
            centers = simulation.particle_pos,
            radius = config.PARTICLE_RADIUS,
            per_vertex_color = simulation.particle_colors,
            index_count = simulation.particle_count[None]
        )

        self.canvas.scene(self.scene)
        self.window.show()

    @property
    def is_running(self):
        return self.window.running

