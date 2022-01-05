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
        
        self.posRight=np.array([0,0,0])
        self.posLeft=np.array([0,0.8,0])
        self.ornRight=p.getQuaternionFromEuler([0,0,0])
        self.ornLeft=p.getQuaternionFromEuler([0,0,0])
        
        right = self.sim.loadURDF("franka_panda/panda.urdf", self.posRight, self.ornRight, useFixedBase=True, flags=flags)
        left = self.sim.loadURDF("franka_panda/panda.urdf", self.posLeft, self.ornLeft, useFixedBase=True, flags=flags)

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

import pybullet_utils.transformations as trans

class DualFrankaPandaTaskTranslationBulletEnv(DualFrankaPandaBulletEnv):
    def __init__(self, args) -> None:
        super().__init__(args)

        #override state and action space for a task representation with only translation component
        #obs: translational position, velocity, note the right basis is taken as the origin
        ws_cubic = np.array([1.0, 1.0, 1.0])
        low_right = np.concatenate([self.posRight - ws_cubic*0.5, -0.2*np.ones(3)])
        low_left = np.concatenate([self.posLeft - ws_cubic*0.5, -0.2*np.ones(3)])
        high_right = np.concatenate([self.posRight + ws_cubic*0.5, 0.2*np.ones(3)])
        high_left = np.concatenate([self.posLeft + ws_cubic*0.5, 0.2*np.ones(3)])
        self.observation_space = spaces.Box(np.concatenate((low_left, low_right)), np.concatenate((high_left, high_right)))

        #act: force
        self.action_space = spaces.Box(-30*np.ones(6), 30*np.ones(6))

        self.com_pos = [0.0]*3
        self.com_pos[2] = 0.04   #Note: this is only valid for franka with hand
        return

    def get_obs(self):
        #taking end-effector positions as the observations, using right basis as the origin
        obs = self.get_ee_pos_and_vel()
        return np.concatenate(obs)

    def get_ee_pos_and_vel(self):
        #return end-effector position and velocities
        arm_ee_state = [self.sim.getLinkState(id,
                        self.pandaNumDofs-1,
                        computeLinkVelocity=1,
                        computeForwardKinematics=0) for id in enumerate(self.robots)]
        arm_ee_trans = [s[0]+s[6] for s in arm_ee_state]

        return arm_ee_trans

    def get_ee_rot(self):
        #return end-effector orientation 
        arm_ee_rot = [self.sim.getLinkState(id,
                self.pandaNumDofs-1,
                computeLinkVelocity=0,
                computeForwardKinematics=0)[1] for id in enumerate(self.robots)]
        return arm_ee_rot
    
    def get_arm_jacobian(self):
        #jacobian w.r.t the joint position at end-effector link
        #note there would be an offset to the real contact point
        zero_vec = [0.0]*self.pandaNumDofs
        #warning: we cannot directly feed numpy array to calculateJacobian. it dumps a segment fault...
        agent_pos = self.get_arm_joint_position()
        jac_lst = [self.sim.calculateJacobian(id, self.pandaNumDofs-1, self.com_pos, agent_pos[i], zero_vec, zero_vec) for i, id in enumerate(self.robots)]
        jac_t_lst = [j[0] for j in jac_lst]
        jac_r_lst = [j[1] for j in jac_lst]
        return jac_t_lst, jac_r_lst

    def get_arm_joint_position(self):
        joint_state = super().get_obs()
        return joint_state[:self.pandaNumDofs], joint_state[2*self.pandaNumDofs:3*self.pandaNumDofs]
    
    def rot_stiffness_control(self, curr_rot, target_rot):
        #err = target_rot * curr_rot^T
        #torque = sign(err[-1])*err[:3]
        err = trans.quaternion_multiply(target_rot, trans.quaternion_conjugate(curr_rot))
        return np.sign(err[-1])*err[:3]
    
    def rot_error(self, curr_rot, target_rot):
        #orientation error expressed in the inertial reference frame
        #note this is different for the one used in stiffness control which is in the body frame
        err = trans.quaternion_multiply(curr_rot, trans.quaternion_conjugate(target_rot))
        return err
    
    def pseudo_inverse_control(self, jac, dx, damp=1e-3):
        #dX = jac(q)dq
        #using damped pseudo inverse to derive dq
        jac_inv = jac.T@np.linalg.pinv(jac@jac.T+damp*np.eye(jac.shape[0]))
        return jac_inv.dot(dx)

    def reset(self):
        super().reset()

        #record desired orientations, quaternion (x,y,z,w)
        self.desiredOrnLeft = np.array(self.sim.getLinkState(self.robots[0],
                        self.pandaNumDofs-1,
                        computeLinkVelocity=1,
                        computeForwardKinematics=0)[1])
        self.desiredOrnRight = np.array(self.sim.getLinkState(self.robots[1],
                        self.pandaNumDofs-1,
                        computeLinkVelocity=1,
                        computeForwardKinematics=0)[1])
        return self.get_obs()
    
    def step(self, a):
        zero_vec = [0.0]*self.pandaNumDofs
        agent_pos = self.get_arm_joint_position()

        agent_ee_rot = self.get_ee_rot()
        
        jac_t_lst, jac_r_lst = self.get_arm_jacobian()

        if self.args.force_ctrl:
            #compute joint actuation from Cartegian forces; use gravity compensation and a fixed stiffness control for the orientation part
            gravity_left = self.sim.calculateInverseDynamics(id, agent_pos[0], zero_vec, zero_vec)
            gravity_right = self.sim.calculateInverseDynamics(id, agent_pos[1], zero_vec, zero_vec)

            rot_control_left = self.rot_stiffness_control(np.array(agent_ee_rot[0]), np.array(self.desiredOrnLeft)).dot(jac_r_lst[0])
            rot_control_right = self.rot_stiffness_control(np.array(agent_ee_rot[1]), np.array(self.desiredOrnRight)).dot(jac_r_lst[1])

            trans_control_left = a[:3].dot(jac_t_lst[0])
            trans_control_right = a[3:].dot(jac_t_lst[1])
            
            super().step(np.concatenate([gravity_left+rot_control_left+trans_control_left, gravity_right+rot_control_right+trans_control_right]))
        else:
            #compute desired joint velocity for Cartesian translational velocity
            #using Jacobian transpose method to reuse the computation for force
            rot_err = [self.rot_error(agent_ee_rot[0], self.desiredOrnLeft), self.rot_error(agent_ee_rot[1], self.desiredOrnRight)]
            
            jacs = [np.concatenate([np.array(jac_t), np.array(jac_r)],axis=0) for jac_t, jac_r in zip(jac_t_lst, jac_r_lst)]
            
            dqs = [self.pseudo_inverse_control(np.array(jac), np.array(err)) for jac, err in zip(jacs, rot_err)]

            super().step(np.concatenate(dqs))
        return 
        