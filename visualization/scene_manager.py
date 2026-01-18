import taichi as ti
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

        """Camera starting position"""
        self.camera.position(1.5, 1.5, 1.5)
        self.camera.lookat(0, 0, 0)

    def render_frame(self, simulation, geometry):
        """
        Render the current frame
        :return:
        """
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        self.scene.point_light(pos=(2, 5, 2), color=config.LIGHT_COLOR)
        self.scene.ambient_light(config.AMBIENT_COLOR)

        ## Drawing voxels
        self.scene.mesh_instance(
            vertices=geometry.verts,
            indices=geometry.indices,
            transforms=simulation.pos,
            instance_count=simulation.count[None],
            color=config.CUBE_COLOR
        )

        self.canvas.scene(self.scene)
        self.window.show()

    @property
    def is_running(self):
        return self.window.running

