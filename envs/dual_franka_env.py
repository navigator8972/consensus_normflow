import os
import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding

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
        self.restPositionsLeft=[-0.98, -1.058, 0, -2.24, 0,  2.5, 0, 0, 0]
        self.restPositionsRight=[0.98, -1.058, 0, -2.24, 0, 2.5, 0, 0, 0]

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
    
    def initialize_bullet(self):
        self.sim.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)  # FEM deform sim
        self.sim_gravity = -10.0
        self.sim.setGravity(0, 0, self.sim_gravity)

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
  
        return

    def reset(self):
        self.initialize_bullet()

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

from .utils import BulletDebugFrameDrawer

class DualFrankaPandaObjectsBulletEnv(DualFrankaPandaBulletEnv):
    """
    Two frankas manipulate two rigidly attached objects and coordinate them to achieve some relative pose
    """
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

        #offset for ignoring hand, 0 for using the last finger
        self.handlinkID_offset = 2

        #overwrite rest position
        self.restPositionsLeft = [-0.14478729478453345, -0.9184881776305733, -0.2018129493638807, -2.631694412176634, -1.7707712904694894, 1.8102247072763602, 2.512651511166867, 0.015, 0.015] #open gripper a bit for avoid collision
        self.restPositionsRight = [0.2584990522331347, -0.6474235673406391, 0.2818125550604209, -1.574409341395127, 0.20990817572737575, 0.9757974029681046, 1.3900018760902164, 0.0, 0.0]

        #time steps of an episode
        self.horizon = 200
        self.t = 0

    def load_objects(self):
        """
        load and attach objects to each Franka hand
        """
        self.objects = [None, None]
        self.objects_cons = [None, None]
        self.objects_mass = [None, None]
        self.objects_com_pos = [None, None]
        self.objects_debug_drawers = [None, None]

        arm_ee_state = [self.sim.getLinkState(id,
                        self.pandaNumJoints-1-self.handlinkID_offset,
                        computeLinkVelocity=1,
                        computeForwardKinematics=1) for id in self.robots]
        
        #find attached link com frame 
        self.attach_link_com_pos = [arm_ee_state[0][2], arm_ee_state[1][2]] #(0.0, 0.01, 0.02)

        #==========================object in the right hand=========================#
        #a cylinder in the right hand
        obj_offset_to_parent_com = [0, -0.01, 0.08]
        #object com offset to attached link joint, for future jacobian computation
        self.objects_com_pos[1] = (np.array(obj_offset_to_parent_com) + np.array(self.attach_link_com_pos[1])).tolist()

        object_pos, object_quat = self.sim.multiplyTransforms(arm_ee_state[1][0], arm_ee_state[1][1], obj_offset_to_parent_com, [0, 0, 0, 1])
        
        cylinder_radius = 0.023
        self.cylinder_height = 0.08
        cylinder_mass = 0.1
        cylinder_colid = self.sim.createCollisionShape(self.sim.GEOM_CYLINDER, radius=cylinder_radius, height=self.cylinder_height)
        self.objects[1] = self.sim.createMultiBody(cylinder_mass, cylinder_colid, basePosition=object_pos, baseOrientation=object_quat)
        
        self.objects_cons[1] = self.sim.createConstraint(self.robots[1], self.pandaNumJoints-1-self.handlinkID_offset, self.objects[1], -1, 
                                self.sim.JOINT_FIXED, 
                                [1, 0, 0],                  #joint axis
                                obj_offset_to_parent_com,   #joint pos relative to parent COM frame
                                [0, 0, 0]                   #joint pos relative to child COM frame
                                )
        self.objects_mass[1] = cylinder_mass

        self.objects_debug_drawers[1] = BulletDebugFrameDrawer(self.sim)

        #==========================object in the left hand=========================#
        #a cup in the left hand
        obj_offset_to_parent_com = [0.01, -0.03, 0.08]
        obj_offset_quat = self.sim.getQuaternionFromEuler([-np.pi/2, 0, -np.pi/2])
        object_pos, object_quat = self.sim.multiplyTransforms(arm_ee_state[0][0], arm_ee_state[0][1], obj_offset_to_parent_com, obj_offset_quat)

        #object com offset to attached link joint, for future jacobian computation
        self.objects_com_pos[0] = (np.array(obj_offset_to_parent_com) + np.array(self.attach_link_com_pos[0])).tolist()

        #local assets
        data_path = os.path.join(os.path.split(__file__)[0], 'assets')
        
        self.sim.setAdditionalSearchPath(data_path)

        self.objects[0] = self.sim.loadURDF('socket.xml', object_pos, object_quat)

        self.sim.changeVisualShape(self.objects[0], -1, rgbaColor=[1, 0, 0, 0.3])

        self.objects_cons[0] = self.sim.createConstraint(self.robots[0], self.pandaNumJoints-1-self.handlinkID_offset, self.objects[0], -1, 
                        self.sim.JOINT_FIXED, 
                        [1, 0, 0],                  #joint axis
                        obj_offset_to_parent_com,   #joint pos relative to parent COM frame
                        [0, 0, 0],                  #joint pos relative to child COM frame
                        obj_offset_quat,            #parentFrameOrientation
                        [0, 0, 0, 1]                #childFrameOrientation
                        )
        #disable collision between mug and gripper
        # self.sim.setCollisionFilterPair(self.objects[0], self.robots[0], -1, self.pandaNumJoints-1, enableCollision=0)
        # self.sim.setCollisionFilterPair(self.objects[0], self.robots[0], -1, self.pandaNumJoints-2, enableCollision=0)
        # self.sim.setCollisionFilterPair(self.objects[0], self.robots[0], -1, self.pandaNumJoints-3, enableCollision=0)

        self.objects_mass[0] = 0.05   #from the URDF

        self.objects_debug_drawers[0] = BulletDebugFrameDrawer(self.sim)
        return
    
    def initialize_task_poses(self):
        #reset initial task poses, can be randomized
        left_pos = [0.25, 0.5, 0.4]
        right_pos = [0.25, 0.24, 0.8]

        left_quat = self.sim.getQuaternionFromEuler([np.pi/2, -np.pi/2, 0])
        right_quat = self.sim.getQuaternionFromEuler([np.pi, 0, 0])

        left_joints = self.sim.calculateInverseKinematics(self.robots[0], self.pandaNumJoints-1, left_pos, left_quat)
        right_joints = self.sim.calculateInverseKinematics(self.robots[1], self.pandaNumJoints-1, right_pos, right_quat)

        # print(left_joints)
        # print(right_joints)

        for i in range(self.pandaNumDofs):
            self.sim.resetJointState(self.robots[0], self.pandaActuatedJointIndices[i], left_joints[i], 0)
            self.sim.resetJointState(self.robots[1], self.pandaActuatedJointIndices[i], right_joints[i], 0)

        return
    
    def reset(self):
        self.initialize_bullet()

        self.initialize_task_poses()

        if self.args.force_ctrl:
            #we need to first set force limit to zero to use torque control!!
            #see: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_dynamics.py
            self.sim.setJointMotorControlArray(self.robots[0], self.pandaActuatedJointIndices[:-2], self.sim.VELOCITY_CONTROL, forces=np.zeros(self.pandaNumDofs-2))
            self.sim.setJointMotorControlArray(self.robots[1], self.pandaActuatedJointIndices[:-2], self.sim.VELOCITY_CONTROL, forces=np.zeros(self.pandaNumDofs-2))
        
        self.load_objects()
        # for i in range(100):
        #     self.sim.stepSimulation()  # step a few steps to get initial state
        
        #record desired orientations, quaternion (x,y,z,w)
        self.desiredOrnLeft = np.array(self.sim.getBasePositionAndOrientation(self.objects[0])[1])
        self.desiredOrnRight = np.array(self.sim.getBasePositionAndOrientation(self.objects[1])[1])

        #reset tick
        self.t = 0

        return self.get_obs()
    
    def get_obs(self):
        #taking end-effector positions as the observations, using right basis as the origin
        obs = self.get_object_pos_and_vel()
        return obs[0]+obs[1]

    def get_object_pos_and_vel(self):
        #return end-effector position and velocities
        object_states = [self.sim.getBasePositionAndOrientation(id) for id in self.objects]
        #we need some extra treatment to cylinder to put interest point at the bottom
        object_states[1] = self.sim.multiplyTransforms(object_states[1][0], object_states[1][1], [0, 0, self.cylinder_height/2], [1, 0, 0, 0])
        #we probably need to account for rotational effects on the bottom of cylider point but if rotational controller is doing a good job...
        object_velocities = [self.sim.getBaseVelocity(id) for id in self.objects]   
        object_trans = [s[0]+v[0] for s, v in zip(object_states, object_velocities)]
        return object_trans

    def get_object_rot(self):
        #return end-effector orientation 
        object_states = [self.sim.getBasePositionAndOrientation(id) for id in self.objects]
        return [s[1] for s in object_states]
    
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

    def get_object_jacobian(self):
        #jacobian w.r.t the joint position at object link
        #note there would be an offset to the com of objects
        zero_vec = [0.0]*(self.pandaNumDofs)
        #warning: we cannot directly feed numpy array to calculateJacobian. it dumps a segment fault...
        left_p, _, _ = self.getMotorJointStates(self.robots[0])
        right_p, _, _ = self.getMotorJointStates(self.robots[1])
        agent_pos = [left_p, right_p]
        ##!! even we are not considering the end-effector link, we need to provide all DOF values
        jac_lst = [self.sim.calculateJacobian(id, self.pandaNumJoints-1-self.handlinkID_offset, self.objects_com_pos[i], agent_pos[i], zero_vec, zero_vec) for i, id in enumerate(self.robots)]
        jac_t_lst = [np.array(j[0])[:, :-self.handlinkID_offset].astype(float) for j in jac_lst]
        jac_r_lst = [np.array(j[1])[:, :-self.handlinkID_offset].astype(float) for j in jac_lst]
        return jac_t_lst, jac_r_lst
    
    def step(self, a):
        #similar to translation task
        #for acting task force at a different jacobian
        left_p, left_v, _ = self.getMotorJointStates(self.robots[0])
        right_p, right_v, _ = self.getMotorJointStates(self.robots[1])
        agent_pos = [left_p, right_p]

        agent_ee_rot = self.get_object_rot()

        #compensate gravity from the object
        jac_t_lst, jac_r_lst = self.get_object_jacobian()

        if self.args.force_ctrl:
            #compute joint actuation from Cartegian forces; use gravity compensation and a fixed stiffness control for the orientation part
            zero_vec = [0.0]*self.pandaNumDofs
            gravity_left = self.sim.calculateInverseDynamics(self.robots[0], agent_pos[0], zero_vec, zero_vec)
            gravity_right = self.sim.calculateInverseDynamics(self.robots[1], agent_pos[1], zero_vec, zero_vec)
            #ignore hand joints
            gravity_left = np.array(gravity_left)[:-2]
            gravity_right = np.array(gravity_right)[:-2]

            objects_gravcomp_joint = [(m*self.sim_gravity*np.array([0, 0, -1])).dot(jac) for jac, m in zip(jac_t_lst, self.objects_mass)]

            rot_control_left = self.rot_stiffness_control(np.array(agent_ee_rot[0]), np.array(self.desiredOrnLeft)).dot(jac_r_lst[0])
            rot_control_right = self.rot_stiffness_control(np.array(agent_ee_rot[1]), np.array(self.desiredOrnRight)).dot(jac_r_lst[1])

            trans_control_left = a[:3].dot(jac_t_lst[0])
            trans_control_right = a[3:].dot(jac_t_lst[1])

            kr = 3   #solely stiffness leads to instability, may need some damping
            kd = 0.1
            #augment some joint damping for stablizing the system
            rot_control_left = kr*rot_control_left - kd*np.array(left_v)[:-2]
            rot_control_right = kr*rot_control_right - kd*np.array(right_v)[:-2]

            
            super().step(np.concatenate([gravity_left+rot_control_left+trans_control_left+objects_gravcomp_joint[0], 
                                        gravity_right+rot_control_right+trans_control_right+objects_gravcomp_joint[1]]))
        else:
            #compute desired joint velocity for Cartesian translational velocity
            #using Jacobian transpose method to reuse the computation for force
            rot_err = [self.rot_error(agent_ee_rot[0], self.desiredOrnLeft), self.rot_error(agent_ee_rot[1], self.desiredOrnRight)]
            errs = [np.concatenate((np.array(a[:3]),rot_err[0][:3])),np.concatenate((np.array(a[3:]),rot_err[1][:3])) ]

            jacs = [np.concatenate([np.array(jac_t), np.array(jac_r)],axis=0) for jac_t, jac_r in zip(jac_t_lst, jac_r_lst)]
            
            dqs = [self.pseudo_inverse_control(jac, err) for jac, err in zip(jacs, errs)]

            super().step(np.concatenate(dqs))
        
        # update object poses in debug
        obs = np.array(self.get_obs())

        if self.args.debug:
            pos = [obs[:3].tolist(), obs[6:9].tolist()]
            quat = self.get_object_rot()

            for d, p, q in zip(self.objects_debug_drawers, pos, quat):
                d.update_drawing(p, q, scale=0.1)

        self.t += 1
        done = False

        #calculate reward as the negative of difference
        running_pos_err = np.linalg.norm(obs[:3]-obs[6:9])  
        running_vel_err = (np.linalg.norm(obs[3:6])+np.linalg.norm(obs[9:]))*0.1    #scale for velocity component
        running_ctrl_efforts = (np.linalg.norm(a[:3])+np.linalg.norm(a[3:]))*0.001  #scale for control penalty
        reward = -running_pos_err-running_vel_err-running_ctrl_efforts

        success = False

        if self.t >= self.horizon:
            terminal_err = np.linalg.norm(obs[:3]-obs[6:9])*10                                                      #scale for the terminal part
            reward -= terminal_err
            if terminal_err < 1e-3:
                success = True

        return obs, reward, done, dict( reward_pos=-running_pos_err, 
                                        reward_vel=-running_vel_err, 
                                        reward_ctrl=-running_ctrl_efforts,
                                        success=success
                                        )
    
