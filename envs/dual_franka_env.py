import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding
from numpy.core.defchararray import array, join

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
        self.sim.setAdditionalSearchPath(pd.getDataPath())

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
        
        #robot base pose
        self.posRight=np.array([0,0,0])
        self.posLeft=np.array([0,0.8,0])
        self.ornRight=p.getQuaternionFromEuler([0,0,0])
        self.ornLeft=p.getQuaternionFromEuler([0,0,0])
        
        #panda arm joint specifications
        # self.pandaNumDofs = 7   #ignore dofs of the gripper
        self.pandaJointLimits = [[-2.8973,  -1.7628,    -2.8973,    -3.0718,    -2.8973,    -0.0175,    -2.8973],
                                 [2.8973,   1.7628,     2.8973,     -0.0698,    2.8973,     3.7525,     2.8973]]
        self.pandaJointVelLimits = [[-2.1750,    -2.1750,     -2.1750,     -2.1750,     -2.6100,     -2.6100,     -2.6100],
                                    [2.1750,    2.1750,     2.1750,     2.1750,     2.6100,     2.6100,     2.6100]]
        self.pandaJointTrqLimits = [[-87,   -87,    -87,    -87,    -12,    -12,    -12],
                                    [87,    87,     87,     87,     12,     12,     12]]

        # self.initJointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
        self.restPositionsLeft=[-0.98, -1.058, 0, -2.24, 0,  2.5, 2.32, 0, 0]
        self.restPositionsRight=[0.98, -1.058, 0, -2.24, 0, 2.5, 2.32, 0, 0]

        self.observation_space = spaces.Box(low=np.array(self.pandaJointLimits[0]+self.pandaJointVelLimits[0]+self.pandaJointLimits[0]+self.pandaJointVelLimits[0]),
                                            high=np.array(self.pandaJointLimits[1]+self.pandaJointVelLimits[1]+self.pandaJointLimits[1]+self.pandaJointVelLimits[1]))
        if args.force_ctrl:
            self.action_space = spaces.Box(low=np.array(self.pandaJointTrqLimits[0]+self.pandaJointTrqLimits[0]),
                                            high=np.array(self.pandaJointTrqLimits[1]+self.pandaJointTrqLimits[1]))
        else:
            self.action_space = spaces.Box(low=np.array(self.pandaJointVelLimits[0]+self.pandaJointVelLimits[0]),
                                            high=np.array(self.pandaJointVelLimits[1]+self.pandaJointVelLimits[1]))
        self.seed()

        return

    def load_robots(self):
        #load right and left arms
        flags = self.sim.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        
        right = self.sim.loadURDF("franka_panda/panda.urdf", self.posRight, self.ornRight, useFixedBase=True, flags=flags)
        left = self.sim.loadURDF("franka_panda/panda.urdf", self.posLeft, self.ornLeft, useFixedBase=True, flags=flags)

        #filter only actuated joint/link index out
        self.pandaNumJoints = self.sim.getNumJoints(left)
        joint_infos = [self.sim.getJointInfo(left, i) for i in range(self.pandaNumJoints)]
        self.pandaActuatedJointIndices = [i for i, info in enumerate(joint_infos) if info[3] > -1]   #note this includes two hand finger prismatic joints and exclude a fixed joint
        self.pandaNumDofs = len(self.pandaActuatedJointIndices)
        #find end-effector com
        result = self.sim.getLinkState(left,
                        self.pandaNumJoints-1,
                        computeLinkVelocity=1,
                        computeForwardKinematics=1)
        self.ee_com_pos = [result[2], result[2]] #COM of the end-effector link

        return [left, right]
    
    def initialize_robot_pose(self):
        #initial joint position
 
        for i in range(self.pandaNumDofs):
            self.sim.resetJointState(self.robots[0], self.pandaActuatedJointIndices[i], self.restPositionsLeft[i], 0)
            self.sim.resetJointState(self.robots[1], self.pandaActuatedJointIndices[i], self.restPositionsRight[i], 0)

        return
    
    def getMotorJointStates(self, robot):
        joint_states = self.sim.getJointStates(robot, self.pandaActuatedJointIndices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques
    
    def reset(self):
        self.sim.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)  # FEM deform sim
        self.sim.setGravity(0, 0, -10.0)

        self.floor_id = self.sim.loadURDF('plane.urdf')
        
        if self.args.viz:  # no rendering during load
            self.sim.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            
        self.robots = self.load_robots()

        self.initialize_robot_pose()

        if self.args.viz:  # loading done, so enable debug rendering if needed
            #time.sleep(0.1)  # wait for debug visualizer to catch up
            self.sim.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            self.sim.resetDebugVisualizerCamera(cameraDistance=self._cam_dist, cameraYaw=self._cam_yaw, cameraPitch=self._cam_pitch, cameraTargetPosition=self._cam_target_pos)

        # for i in range(self.pandaNumDofs):    #ignore two finger joints
        #     self.sim.setJointMotorControl2(self.robots[0], self.pandaActuatedJointIndices[i], self.sim.POSITION_CONTROL, self.restPositionsLeft[i],force=5 * 240.)
        #     self.sim.setJointMotorControl2(self.robots[1], self.pandaActuatedJointIndices[i], self.sim.POSITION_CONTROL, self.restPositionsRight[i],force=5 * 240.)
  
        # for i in range(100):
        #     self.sim.stepSimulation()  # step a few steps to get initial state

        obs = self.get_obs()
        return obs
    
    def get_obs(self):
        # left_states = self.sim.getJointStates(self.robots[0], range(self.pandaNumDofs))
        # right_states = self.sim.getJointStates(self.robots[1], range(self.pandaNumDofs))
        # obs = []
        # for i in range(self.pandaNumDofs):
        #     obs.append(list(left_states[i][:2]+right_states[i][:2]))
        # obs = np.array(obs).T.flatten()
        left_p, left_v, _ = self.getMotorJointStates(self.robots[0])
        right_p, right_v, _ = self.getMotorJointStates(self.robots[1])
        return left_p[:self.pandaNumDofs-2]+left_v[:self.pandaNumDofs-2]+right_p[:self.pandaNumDofs-2]+right_v[:self.pandaNumDofs-2]
    
    def step(self, a):
        if self.args.force_ctrl:
            for i in range(self.pandaNumDofs-2):
                self.sim.setJointMotorControl2(self.robots[0], self.pandaActuatedJointIndices[i], self.sim.TORQUE_CONTROL, force=a[i])
                self.sim.setJointMotorControl2(self.robots[1], self.pandaActuatedJointIndices[i], self.sim.TORQUE_CONTROL, force=a[i+self.pandaNumDofs-2])
        else:
            for i in range(self.pandaNumDofs-2):
                self.sim.setJointMotorControl2(self.robots[0], self.pandaActuatedJointIndices[i], self.sim.VELOCITY_CONTROL, targetVelocity=a[i], force=5 * 240.)
                self.sim.setJointMotorControl2(self.robots[1], self.pandaActuatedJointIndices[i], self.sim.VELOCITY_CONTROL, targetVelocity=a[i+self.pandaNumDofs-2], force=5 * 240.)
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

        return

    def get_obs(self):
        #taking end-effector positions as the observations, using right basis as the origin
        obs = self.get_ee_pos_and_vel()
        return obs[0]+obs[1]

    def get_ee_pos_and_vel(self):
        #return end-effector position and velocities
        arm_ee_state = [self.sim.getLinkState(id,
                        self.pandaNumJoints-1,
                        computeLinkVelocity=1,
                        computeForwardKinematics=0) for id in self.robots]
        arm_ee_trans = [s[0]+s[6] for s in arm_ee_state]

        return arm_ee_trans

    def get_ee_rot(self):
        #return end-effector orientation 
        arm_ee_rot = [self.sim.getLinkState(id,
                self.pandaNumJoints-1,
                computeLinkVelocity=0,
                computeForwardKinematics=0)[1] for id in self.robots]
        return arm_ee_rot
    
    def get_arm_jacobian(self):
        #jacobian w.r.t the joint position at end-effector link
        #note there would be an offset to the real contact point
        zero_vec = [0.0]*(self.pandaNumDofs)
        #warning: we cannot directly feed numpy array to calculateJacobian. it dumps a segment fault...
        left_p, _, _ = self.getMotorJointStates(self.robots[0])
        right_p, _, _ = self.getMotorJointStates(self.robots[1])
        agent_pos = [left_p, right_p]
        jac_lst = [self.sim.calculateJacobian(id, self.pandaNumJoints-1, self.ee_com_pos[i], agent_pos[i], zero_vec, zero_vec) for i, id in enumerate(self.robots)]
        jac_t_lst = [np.array(j[0])[:, :-2].astype(float) for j in jac_lst]
        jac_r_lst = [np.array(j[1])[:, :-2].astype(float) for j in jac_lst]
        return jac_t_lst, jac_r_lst

    def get_arm_joint_position(self):
        left_p, _, _ = self.getMotorJointStates(self.robots[0])
        right_p, _, _ = self.getMotorJointStates(self.robots[1])
        return left_p[:-2], right_p[:-2]    #ignore two hand joints
    
    def rot_stiffness_control(self, curr_rot, target_rot):
        #err = target_rot * curr_rot^T
        #torque = sign(err[-1])*err[:3]
        # err = trans.quaternion_multiply(target_rot, trans.quaternion_conjugate(curr_rot))
        err = self.rot_error(target_rot, curr_rot)
        return np.sign(err[-1])*err[:3]
    
    def rot_error(self, target_rot, curr_rot):
        #orientation error expressed in the inertial reference frame
        err = trans.quaternion_multiply(target_rot, trans.quaternion_conjugate(curr_rot))
        return err
    
    def pseudo_inverse_control(self, jac, dx, damp=1e-3):
        #dX = jac(q)dq
        #using damped pseudo inverse to derive dq
        # print(jac.shape, dx.shape)
        jac_inv = jac.T@np.linalg.pinv(jac@jac.T+damp*np.eye(jac.shape[0]))
        return -jac_inv.dot(dx)

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
        if self.args.force_ctrl:
            #we need to first set force limit to zero to use torque control!!
            #see: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_dynamics.py
            self.sim.setJointMotorControlArray(self.robots[0], self.pandaActuatedJointIndices[:-2], self.sim.VELOCITY_CONTROL, forces=np.zeros(self.pandaNumDofs-2))
            self.sim.setJointMotorControlArray(self.robots[1], self.pandaActuatedJointIndices[:-2], self.sim.VELOCITY_CONTROL, forces=np.zeros(self.pandaNumDofs-2))

        return self.get_obs()
    
    def step(self, a):
        zero_vec = [0.0]*self.pandaNumDofs
        left_p, left_v, _ = self.getMotorJointStates(self.robots[0])
        right_p, right_v, _ = self.getMotorJointStates(self.robots[1])
        agent_pos = [left_p, right_p]

        agent_ee_rot = self.get_ee_rot()
        
        jac_t_lst, jac_r_lst = self.get_arm_jacobian()

        if self.args.force_ctrl:
            #compute joint actuation from Cartegian forces; use gravity compensation and a fixed stiffness control for the orientation part
            gravity_left = self.sim.calculateInverseDynamics(self.robots[0], agent_pos[0], zero_vec, zero_vec)
            gravity_right = self.sim.calculateInverseDynamics(self.robots[1], agent_pos[1], zero_vec, zero_vec)
            #ignore hand joints
            gravity_left = np.array(gravity_left)[:-2]
            gravity_right = np.array(gravity_right)[:-2]

            rot_control_left = self.rot_stiffness_control(np.array(agent_ee_rot[0]), np.array(self.desiredOrnLeft)).dot(jac_r_lst[0])
            rot_control_right = self.rot_stiffness_control(np.array(agent_ee_rot[1]), np.array(self.desiredOrnRight)).dot(jac_r_lst[1])

            trans_control_left = a[:3].dot(jac_t_lst[0])
            trans_control_right = a[3:].dot(jac_t_lst[1])

            kr = 1   #1 leads to instability...
            kd = 0.1
            #augment some joint damping for stablizing the system
            rot_control_left = kr*rot_control_left - kd*np.array(left_v)[:-2]
            rot_control_right = kr*rot_control_right - kd*np.array(right_v)[:-2]

            
            super().step(np.concatenate([gravity_left+rot_control_left+trans_control_left, gravity_right+rot_control_right+trans_control_right]))
        else:
            #compute desired joint velocity for Cartesian translational velocity
            #using Jacobian transpose method to reuse the computation for force
            rot_err = [self.rot_error(agent_ee_rot[0], self.desiredOrnLeft), self.rot_error(agent_ee_rot[1], self.desiredOrnRight)]
            errs = [np.concatenate((np.array(a[:3]),rot_err[0][:3])),np.concatenate((np.array(a[3:]),rot_err[1][:3])) ]

            jacs = [np.concatenate([np.array(jac_t), np.array(jac_r)],axis=0) for jac_t, jac_r in zip(jac_t_lst, jac_r_lst)]
            
            dqs = [self.pseudo_inverse_control(jac, err) for jac, err in zip(jacs, errs)]

            super().step(np.concatenate(dqs))
        return 

class DualFrankaPandaAssemblyBulletEnv(DualFrankaPandaTaskTranslationBulletEnv):
    """
    Two frankas manipulate two rigidly attached objects and assemble them to achieve some relative pose
    """
    def __init__(self, args) -> None:
        super().__init__(args)
    

    def load_objects(self):
        """
        load and attach objects to each Franka hand
        """
        arm_ee_state = [self.sim.getLinkState(id,
                        self.pandaNumJoints-1,
                        computeLinkVelocity=1,
                        computeForwardKinematics=0) for id in self.robots]

        #a cylinder in the right hand
        cylinder_radius = 0.023
        cylinder_height = 0.08
        cylinder_colid = self.sim.createCollisionShape(self.sim.GEOM_CYLINDER, radius=cylinder_radius, height=cylinder_height)
        self.cylinder = self.sim.createMultiBody(0.1, cylinder_colid, basePosition=arm_ee_state[1][0], baseOrientation=arm_ee_state[1][1])
        
        #a cup in the left hand
        self.cup = self.sim.loadURDF('objects/mug.urdf', arm_ee_state[0][0], arm_ee_state[0][1])

        self.sim.changeVisualShape(self.cup, -1, rgbaColor=[1, 0, 0, 0.3])
        
        return
    
    def reset(self):
        super().reset()

        #randomize initial pose here

        #load objects and attach them
        self.load_objects()
        #this is probably a bad idea since we need extra care to the end-effector link jacobian and gravity compensation
        #another idea is using URDFEditor to join the object to the robot, we need to overload load_robots and reset
        #and prepare URDF files for two objects
        #see https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_utils/examples/combineUrdf.py
        return self.get_obs()
    
    def get_obs(self):
        #TODO may need an offset from the raw end-effectors
        return super().get_obs()