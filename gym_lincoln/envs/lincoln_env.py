import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np

class LincolnEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """"
        Description:
            A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
        Source:
            This environment corresponds to the version of dynamic positioning as described by Nick Kougiatsos
        Observation: 
            Type: Box(6)
            Num	Observation                 Min         Max
            0	Y position                  -4 m         4 m
            1	Y speed                     -Inf         Inf 
            2   Y acc                       -Inf         Inf
            3	Heading angle              -5 deg        5 deg
            4	Yaw speed                   -Inf         Inf
            5	Yaw acc                     -Inf         Inf
            
        Actions:
            Type: Discrete(2)
            Num	Action
            0	side+
            1   side-
            2   ccw
            3   cw
            
            Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
        Reward:
            Reward is 1 for every step taken, including the termination step
        Starting State:
            All observations are assigned a uniform random value in [-0.05..0.05]
        Episode Termination:
            Heading angle exceeds 5 deg from start
            Y position goes beyond 2.5 m from start position
            Episode length is greater than 200
            Solved Requirements
            Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
        """
        self.tau=0.02
        self.max_speed=5*0.5144
        self.T1=383.22
        self.T2=292.01
        self.torque_scale=0.0547
        self.force_scale=0.0993
        self.max_yaw_speed=math.pi/12
        self.g=9.81
        self.L=4.1
        self.B=2
        self.T=0.15
        self.CB=0.6
        self.m=180*1.2+23*1.1+3*70+12*2+80
        self.LCG=0.5
        self.dense=1025
        self.Iz=self.m*(0.25*self.L**2)-self.m*self.LCG**2
        self.viewer = None
        self.max_torque=30*self.g*2.95*self.torque_scale/2
        self.max_force=30*self.g*self.force_scale
        self.x_threshold=2.5
        self.y_threshold=2.5
        self.yaw_threshold_radians = 5*math.pi / 360
        high = np.array([self.y_threshold, np.finfo(np.float32).max, np.finfo(np.float32).max, self.yaw_threshold_radians, np.finfo(np.float32).max, np.finfo(np.float32).max],dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.kinematics_integrator = 'euler'
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.seed()
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]  

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        y, y_dot, y_2dot, h, h_dot, h_2dot = state
        if (action==0):
            force = self.max_force
            torque=0
        elif (action==1):
            force = -self.max_force
            torque=0
        elif (action==2):
            force = 0
            torque=self.max_torque
        else:
            force = 0
            torque=-self.max_torque

        if self.kinematics_integrator == 'euler':
            h_2dot = h_2dot + self.tau * (-self.T2*h_2dot-h+torque)/self.T1
            h_dot = h_dot + self.tau * h_2dot
            h  = h + self.tau * h_dot
            y_2dot = y_2dot + self.tau * (-self.T2*y_2dot-y+force)/self.T1
            y_dot = y_dot + self.tau * y_2dot
            y  = y + self.tau * y_dot

        self.state = (y, y_dot, y_2dot, h, h_dot, h_2dot)
        done =  y < -self.y_threshold \
            or y > self.y_threshold \
            or h < -self.yaw_threshold_radians  \
            or h > self.yaw_threshold_radians 
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Dynamic instability!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 400

        world_height = self.y_threshold*2
        scale = screen_height/world_height
        shiplength = self.L*20
        shipwidth = self.B*20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -shiplength/2, shiplength/2, shipwidth/2, -shipwidth/2
            ship= rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            ship.add_attr(rendering.Transform(translation=(screen_width/2,0)))
            self.shiptrans = rendering.Transform()
            ship.add_attr(self.shiptrans)
            self.viewer.add_geom(ship)
            

        if self.state is None: return None

        x = self.state
        carty = x[0]*scale+screen_height/2.0 # MIDDLE OF CART
        theta=x[3]
        self.shiptrans.set_translation(0, carty)
        self.shiptrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None