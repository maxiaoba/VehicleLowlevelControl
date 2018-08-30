const CAR_LENGTH = 4.0
const CAR_WIDTH = 1.5
const AXLE_DISTANCE = 2.66
const CAR_A = 1.0 # distance between cg and front axle [m]
const CAR_B = 1.66 # distance between cg and rear axle [m]

const THETA_MAX = pi/4
const DECELERATION_MAX = 3.0
const ACCELERATION_MAX = 3.0
const STEER_MAX = 25/180*pi #15/180*pi #curvature limit is 0.125 atan(0.125/AXLE_DISTANCE) about 18  #25/180*pi

const DACC_MAX = ACCELERATION_MAX*0.3
const DSTEER_MAX = STEER_MAX*0.3

const STEER_DOT_MAX = 45/180*pi
const ACC_LAT_MAX = 0.3*9.8
const MINIMUM_SPEED = 0.0
const MAXIMUM_SPEED = 5.0
const DESIRE_SPEED = 5.0
const S_MIN = 3*CAR_LENGTH #minimum gap in IDM model

const IPOPT_MAX_CPU_TIME = 0.05

const LANE_WIDTH = 3.0
const LANE_NUM = 2
const EGO_START_X = 100.0
const TOTALSTEP = 200
const TIMESTEP = 0.05

const Y_NOISE = 0.3
const V_NOISE = 0.3
const THETA_NOISE = 10/190*pi

const ACONST = [-8.62167 -32.8740; 1.0 0.0]
const BCONST = [1.0; 0.0]
const CCONST = [0.0 32.7687;]
