import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding
from numpy.core.defchararray import array

import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bclient

class DualFrankaPandaBulletEnv(gym.Env):
    """
    An environment of two franka panda robots for a dual arm setup
    Observation and action in the joint space
    obs: pos + vel
    act: vel/trq
    """
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.sim = bclient.BulletClient(connection_mode=p.GUI if args.viz else p.DIRECT)

        #for the visualizer
        self._cam_dist = 1
        self._cam_yaw = 90
        self._cam_pitch=-30
        self._cam_roll=0
        self._cam_target_pos = [0.25, 0.5, 0.5]
        self._cam_res = [256, 256]
        self.sim.resetDebugVisualizerCamera(cameraDistance=self._cam_dist, cameraYaw=self._cam_yaw, cameraPitch=self._cam_pitch, cameraTargetPosition=self._cam_target_pos)
        self._cam_mat = self.sim.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._cam_target_pos, distance=self._cam_dist, yaw=self._cam_yaw, pitch=self._cam_pitch, roll=self._cam_roll, upAxisIndex=2
        )
        self._cam_proj_mat = [1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, -1.0000200271606445, -1.0,
                         0.0, 0.0, -0.02000020071864128, 0.0]
        
        #panda arm joint specifications
        self.pandaNumDofs = 7
        self.pandaJointLimits = [[-2.8973,  -1.7628,    -2.8973,    -3.0718,    -2.8973,    -0.0175,    -2.8973],
                                 [2.8973,   1.7628,     2.8973,     -0.0698,    2.8973,     3.7525,     2.8973]]
        self.pandaJointVelLimits = [[-2.1750,    -2.1750,     -2.1750,     -2.1750,     -2.6100,     -2.6100,     -2.6100],
                                    [2.1750,    2.1750,     2.1750,     2.1750,     2.6100,     2.6100,     2.6100]]
        self.pandaJointTrqLimits = [[-87,   -87,    -87,    -87,    -12,    -12,    -12],
                                    [87,    87,     87,     87,     12,     12,     12]]
        
        self.observation_space = spaces.Box(low=np.array(self.pandaJointLimits[0]+self.pandaJointVelLimits[0]+self.pandaJointLimits[0]+self.pandaJointVelLimits[0]),
                                            high=np.array(self.pandaJointLimits[1]+self.pandaJointVelLimits[1]+self.pandaJointLimits[1]+self.pandaJointVelLimits[1]))
        if args.force_ctrl:
            self.action_space = spaces.Box(low=np.array(self.pandaJointTrqLimits[0]+self.pandaJointTrqLimits[0]),
                                            high=np.array(self.pandaJointTrqLimits[1]+self.pandaJointTrqLimits[1]))
        else:
            self.action_space = spaces.Box(low=np.array(self.pandaJointVelLimits[0]+self.pandaJointVelLimits[0]),
                                            high=np.array(self.pandaJointVelLimits[1]+self.pandaJointVelLimits[1]))
        self.seed()

    def load_robots(self, args):
        #load right and left arms
        self.sim.setAdditionalSearchPath(pd.getDataPath())
        flags = self.sim.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        ornRight=p.getQuaternionFromEuler([0,0,0])
        ornLeft=p.getQuaternionFromEuler([0,0,0])
        right = self.sim.loadURDF("franka_panda/panda.urdf", np.array([0,0,0]), ornRight, useFixedBase=True, flags=flags)
        left = self.sim.loadURDF("franka_panda/panda.urdf", np.array([0,0.8,0]), ornLeft, useFixedBase=True, flags=flags)

        #initial joint position
        # self.initJointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
        self.restPositionsLeft=[-0.98, -1.058, 0, -2.24, 0,  2.5, 2.32]
        self.restPositionsRight=[0.98, -1.058, 0, -2.24, 0, 2.5, 2.32]
        for i in range(self.pandaNumDofs):
            self.sim.setJointMotorControl2(left, i, self.sim.POSITION_CONTROL, self.restPositionsLeft[i],force=5 * 240.)
            self.sim.setJointMotorControl2(right, i, self.sim.POSITION_CONTROL, self.restPositionsRight[i],force=5 * 240.)

        return [left, right]
    
    def reset(self):
        self.sim.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)  # FEM deform sim
        
        if self.args.viz:  # no rendering during load
            self.sim.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            
        self.robots = self.load_robots(self.args)

        if self.args.viz:  # loading done, so enable debug rendering if needed
            #time.sleep(0.1)  # wait for debug visualizer to catch up
            self.sim.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            self.sim.resetDebugVisualizerCamera(cameraDistance=self._cam_dist, cameraYaw=self._cam_yaw, cameraPitch=self._cam_pitch, cameraTargetPosition=self._cam_target_pos)

        for i in range(100):
            self.sim.stepSimulation()  # step a few steps to get initial state

        obs = self.get_obs()
        return obs
    
    def get_obs(self):
        left_states = self.sim.getJointStates(self.robots[0], range(self.pandaNumDofs))
        right_states = self.sim.getJointStates(self.robots[1], range(self.pandaNumDofs))
        obs = []
        for i in range(self.pandaNumDofs):
            obs.append(list(left_states[i][:2]+right_states[i][:2]))
        obs = np.array(obs).T.flatten()
        return obs
    
    def step(self, a):
        if self.args.force_ctrl:
            for i in range(self.pandaNumDofs):
                self.sim.setJointMotorControl2(self.robots[0], i, self.sim.TORQUE_CONTROL, force=a[i])
                self.sim.setJointMotorControl2(self.robots[1], i, self.sim.TORQUE_CONTROL, force=a[i+self.pandaNumDofs])
        else:
            for i in range(self.pandaNumDofs):
                self.sim.setJointMotorControl2(self.robots[0], i, self.sim.VELOCITY_CONTROL, targetVelocity=a[i], force=5 * 240.)
                self.sim.setJointMotorControl2(self.robots[1], i, self.sim.VELOCITY_CONTROL, targetVelocity=a[i+self.pandaNumDofs], force=5 * 240.)
        self.sim.stepSimulation()
        return

    def render(self, mode='rgb_array'):
        (_, _, px, _, _) = self.sim.getCameraImage(width=self._cam_res[0],
                                              height=self._cam_res[1],
                                              viewMatrix=self._cam_mat,
                                              projectionMatrix=self._cam_proj_mat,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self._cam_res[0],self._cam_res[1], 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array