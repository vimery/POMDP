#!/usr/bin/env python3


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

try:
    import carla
    from carla import ColorConverter as Cc
    from agents.basic_agent import BasicAgent
except ImportError:
    raise RuntimeError('cannot import carla, please ensure carla is installed!')

import argparse
import random
import weakref

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, please ensure pygame is installed!')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, hud, actor_filter, actor_role_name='ego'):
        self.world = carla_world  # carla world
        self.actor_role_name = actor_role_name
        self.hud = hud  # display surface
        self.camera_manager = None  # sensors
        self._actor_filter = actor_filter  # filter of actor
        self.map = self.world.get_map()
        self.player = None
        self.restart()  # generate player
        self.recording_enabled = False  # record flag

    def restart(self):
        # keep the camera config
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # spawn the player
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # set up sensors
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.player
        ]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))
        ]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', Cc.Raw, 'Camera RGB']
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self.index = None

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            # remove alpha
            array = array[:, :, :3]
            # reverse
            array = array[:, :, ::-1]
            # create surface
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame_number)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)

    def tick(self, world, clock):
        pass


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    world = None
    client = None
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        client.start_recorder("recording.log")

        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args.filter, args.rolename)

        agent = BasicAgent(world.player)
        spawn_point = world.map.get_spawn_points()[0]
        agent.set_destination((spawn_point.location.x,
                               spawn_point.location.y,
                               spawn_point.location.z))
        # agent.set_destination((0, 0, 0))
        # print(agent._local_planner._waypoints_queue)

        clock = pygame.time.Clock()
        while True:
            # lock the frame rate to 60 fps
            clock.tick_busy_loop(60)
            world.tick(clock)
            world.render(display)
            # update surface
            pygame.display.flip()
            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)
            # world.player.set_autopilot(True)
    finally:
        if world is not None:
            world.destroy()
        if client is not None:
            client.stop_recorder()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(description='add a car to the world')
    argparser.add_argument('--host', default='127.0.0.1', metavar='H',
                           help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', default='2000', metavar='P', type=int,
                           help='TCP port of the host server (default: 2000)')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720',
                           help='window resolution (default 1280x720)')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*',
                           help='actor filter (default： "vehicle.*")')
    argparser.add_argument('--rolename', metavar='NAME', default='ego',
                           help='actor role name (default: "ego")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]
    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancled by user. Bye!')


if __name__ == '__main__':
    main()
