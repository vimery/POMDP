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

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(2.0)
client.replay_file("recording.log", 0, 0, 0)
