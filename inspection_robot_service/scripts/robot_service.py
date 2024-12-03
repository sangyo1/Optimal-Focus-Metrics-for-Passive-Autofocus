#!/usr/bin/env python3
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, curve_fit
import pandas as pd

# ROS Imports
import rospy
import actionlib
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
import rosbag
                                                        
# Message Imports
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from moveit_msgs.msg import ExecuteTrajectoryAction, MoveGroupActionResult, ExecuteTrajectoryActionResult
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from control_msgs.msg import FollowJointTrajectoryActionFeedback
from inspection_msgs.msg import Inspect

# Service Imports
from inspection_vision.srv import GetSharpness
from inspection_task_planning.srv import InspectionTrigger, InspectionTriggerRequest
from inspection_srvs.srv import Calibrate, CalibrateResponse

# Action Imports
from inspection_task_planning.msg import InspectionAction, InspectionGoal, InspectionFeedback

# MoveIt Imports                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

WAITING = 0
READY = 1

class InspectionRobotService(object):
                         

    def __init__(self):
        rospy.init_node('inspection_robot_service')
        self.rate = rospy.Rate(60)

        self.waiting = False
        self.image_dict = {}

        # -----------------------------------------------------                                                
        # INSPECTION ACTION PLANNING

        # Connect to Inspection Action Planning Server and Trigger
        rospy.loginfo("Connecting to /inspection/action_planning action server...")
        self.inspection_client = actionlib.SimpleActionClient('/inspection/action_planning', InspectionAction)
        rospy.loginfo("Connecting to /inspection/action_planning action server...") 
        self.inspection_client.wait_for_server()
        rospy.loginfo("Connecting to /inspection/action_planning action server...") 
        self.inspection_trigger = rospy.ServiceProxy('/inspection/trigger', InspectionTrigger)
        er = rospy.ServiceProxy('/inspection/trigger', InspectionTrigger)
        rospy.loginfo("Connecting to /inspection/action_planning action server...") 
        self.inspection_trigger.wait_for_service()
        rospy.loginfo("Connected to /inspection/action_planning action server.")

        # Cancel any and all goals currently being executed
        rospy.loginfo(f'/inspection/action_planning state: {self.inspection_client.get_goal_status_text()}')
        self.inspection_client.cancel_all_goals()

        # Begin inspection
        start_inspection = InspectionGoal()
        self.inspection_client.send_goal(start_inspection, feedback_cb=self.inspection_feedback_callback)
        rospy.loginfo(f'/inspection/action_planning state: {self.inspection_client.get_goal_status_text()}')

        # Autofocus
        rospy.loginfo("Waiting for /inspection/perception/sharpness server.")
        rospy.wait_for_service('/inspection/perception/sharpness')
        self.focus_client = rospy.ServiceProxy('/inspection/perception/sharpness', GetSharpness)
        self.autofocus_pub = rospy.Publisher('/inspection/robot/focus', Float64, queue_size=10)


        # -----------------------------------------------------
        # INSPECTION VIEWPOINTS

        self.state = WAITING
        self.inspection_poses = None # List of poses in 'world' coordinate frame we will visit during inspection.

        # TODO: Subscribe to /inspection/auto_partitioning/poses and implement auto_partitioning_callback() function to
        #       save poses in world coordinate frame.
        self.inspection_queue = []
        self.inspection_dict = {}
        self.inspection_sub = rospy.Subscriber("/inspection/viewpoint/inspect", Inspect, self.queue_viewpoint)
        self.inspection_pose_sub = rospy.Subscriber("/inspection/viewpoints", PoseArray, self.auto_partitioning_callback)

        # -----------------------------------------------------
        # ROBOT MOVEIT

        rospy.loginfo("Connecting to robot...")
        moveit_commander.roscpp_initialize(sys.argv)

        self.sim_robot = moveit_commander.RobotCommander() #self.robot = moveit_commander.RobotCommander() 
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        self.move_group.set_max_velocity_scaling_factor(0.02)
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                        moveit_msgs.msg.DisplayTrajectory,
                                                        queue_size=20)

        # TF2 Listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.pose_pub = rospy.Publisher('/pose', PoseStamped, queue_size=10)

        # Cartesian path planning settings
        self.eef_step = 0.0001
        self.jump_threshold = 0.0
        self.speed_scaling = 1.0

        # Home Joint States
        self.home_joints = [-2.9766157309161585, -1.941446443597311, -2.02421236038208, -0.12417157114062505, 1.4312734603881836, -2.988797728215353]

        # Simulation parameters
        if rospy.has_param('~sim_robot'):
            self.sim_robot = rospy.get_param('~sim_robot')
            rospy.loginfo("Got sim_robot param")
        if rospy.has_param('~sim_camera'):
            self.sim_camera = rospy.get_param('~sim_camera')
            rospy.loginfo("Got sim_camera param")
        if rospy.has_param('~part_num'):
            self.part_num = rospy.get_param('~part_num')
            rospy.loginfo("Got part_num param")

        if self.sim_robot:
            self.move_group.set_max_velocity_scaling_factor(1)
        
        # -----------------------------------------------------
        # AUTO-FOCUS 

        self.autofocus_data = []
        self.z_off_pub = rospy.Publisher('/inspection/autofocus/z_off', Float64, queue_size=10)
        self.bag_file_path = f'{str(Path.home())}/inspection/rosbag_data/autofocus_data.bag'

        self.sharpness = 0
        self.sharpness_sub = rospy.Subscriber('/inspection/perception/sobel', Float64, self.sharpness_callback)

        self.pose_opt = None
        self.z_off_opt = 0
        self.z_off_opt_candidate = 0
        self.sharpness_opt = 0
        
        # For measuring sharpness
        self.moving = False
        self.ex_traj_result_sub = rospy.Subscriber('/execute_trajectory/result', ExecuteTrajectoryActionResult, self.ex_traj_result_callback)

        self.offset = 0
        self.tf_wt = self.tfBuffer.lookup_transform('world', 
                                                    'inspection_camera_frame',
                                                    rospy.Time(0),
                                                    rospy.Duration(1.0))

        # -----------------------------------------------------
        # CALIBRATION

        self.calibration_service = rospy.Service('/inspection/calibration', Calibrate, self.calibration_callback)

                
        
    # -----------------------------------------------------
    # INSPECTION VIEWPOINTS

    def queue_viewpoint(self, inspect_msg):
        self.inspection_queue.append(inspect_msg)
        
    # -----------------------------------------------------
    # ROBOT SERVICE

    
    def ex_traj_result_callback(self, msg):
        #print("Hello world.")
        self.moving = False

    def sharpness_callback(self, msg):
        self.sharpness = msg.data

    def run(self):
        while not rospy.is_shutdown():
            # If there is a pose in queue, inspect.
            if self.inspection_queue:
                inspect_msg = self.inspection_queue.pop(0)
                self.inspect(inspect_msg)
            
    def inspect(self, inspect_msg):
        """ Move robot to viewpoint pose, focus, and capture image """

        start_time = time.time()
        #self.move_to_home_position
        # If viewpoint has valid frame_id, move to pose
        print(inspect_msg.viewpoint.header.frame_id)
        if inspect_msg.viewpoint.header.frame_id:
            rospy.loginfo(f'Moving to {inspect_msg.name}...')
            viewpoint_frame_to_world = self.tfBuffer.lookup_transform('world', inspect_msg.viewpoint.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            pose = tf2_geometry_msgs.do_transform_pose(inspect_msg.viewpoint, viewpoint_frame_to_world).pose
            self.move_group.set_max_velocity_scaling_factor(0.5) 
            self.move_group.set_pose_target(pose)
            self.move_group.go(wait=True)

        if inspect_msg.autofocus:
            rospy.loginfo('Focusing...')
            #self.init_autofocus_calibration(z_off_init=inspect_msg.z_off_init)
            self.autofocus(z_off_init=inspect_msg.z_off_init)
            # self.original_position() #for testing with focus, check it is move back to originial position
        
        if inspect_msg.capture:
            rospy.loginfo('Capturing image...')
            self.inspection_dict[inspect_msg.name] = inspect_msg
            self.trigger(inspect_msg.name)
            # add to back to original start position
            #self.original_position()
            

        end_time = time.time()
        rospy.loginfo(f'Done! Took {end_time - start_time} seconds')
       
    def quit(self):
        rospy.loginfo('Cancelling inspection goal...')
        self.inspection_client.cancel_all_goals()

    # -----------------------------------------------------
    # INSPECTION POSES

    def auto_partitioning_callback(self, msg):
        # Look up transform from part to world frame.
        tf_part_to_world = self.tfBuffer.lookup_transform('world', msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))

        # Iterate through msg.poses, transforming into 'world' frame.
        # Save all 'world' frame PoseStamped poses in list 
        self.inspection_poses = []
        pose_list=msg.poses

        for i in range(len(pose_list)):
            ps_p = PoseStamped()
            ps_p.header = msg.header
            ps_p.pose = msg.poses[i]
            ps_w = tf2_geometry_msgs.do_transform_pose(ps_p, tf_part_to_world)
            self.inspection_poses.append(ps_w)        

        self.state = READY


    # -----------------------------------------------------
    # INSPECTION REGIMES

    # -----------------------------------------------------
    # ROBOT MOVEMENT

    def scale_trajectory_time(self, robot_trajectory):
        """ Scale time_from_start by scaling factor for each point in RobotTrajectory. """
        #print(f'Initial duration: {robot_trajectory.joint_trajectory.points[-1].time_from_start}')
        for i in range(len(robot_trajectory.joint_trajectory.points)):
            robot_trajectory.joint_trajectory.points[i].time_from_start = 1/self.speed_scaling * robot_trajectory.joint_trajectory.points[i].time_from_start
        #print(f'Final duration: {robot_trajectory.joint_trajectory.points[-1].time_from_start}')
        return robot_trajectory
    
    def move_to_home_position(self):
        rospy.loginfo("Moving to home position...")
        self.move_group.set_joint_value_target(self.home_joints)
        self.move_group.go(wait=True)
        
           

    def autofocus(self, z_off_init=0.0):
        self.autofocus_data = []
        # look up current joint position move.get.joint something like  moveit_commander.move_group.MoveGroupCommander.get_current_joint_values 	
        self.original_joint_positions = self.move_group.get_current_joint_values()
        # Look up the transform to current location of eef
        # transform: inspection_camera_frame to world
        self.move_group.set_max_velocity_scaling_factor(1) 
        tf_tw_init = self.tfBuffer.lookup_transform('world', 
                                                'inspection_camera_frame',
                                                rospy.Time(0),
                                                rospy.Duration(1.0))
        #print(f'tf_init = {tf_tw_init}')
        # transform: world to inspection_camera_frame, used for logging offset data
        self.tf_wt = self.tfBuffer.lookup_transform('inspection_camera_frame', 
                                                'world',
                                                rospy.Time(0),
                                                rospy.Duration(1.0))
        # Set the baseline variables for checking focus
        self.sharpness_opt = 0
        self.pose_opt = None
        self.z_off_opt = 0
        # rospy.loginfo(f'Moving to z_off_init = {z_off_init}...')
        # self.translate_eef_frame_wrt(tf_tw_init, [0, 0, z_off_init])

        # Set the maximum displacement for auto-focus
        z_off_l = -0.01 # units: meters
        z_off_h = 0.01
        n = 1
        max_off = 0.1
        
        threshold_sharpness = 200
        self.speed_scaling = 1.0
        z_off_arr_init = np.linspace(self.z_off_opt_candidate, self.z_off_opt_candidate, n)
        path_init = np.zeros((n, 3))
        path_init[:,2] = z_off_arr_init
        #self.z_off_opt_candidate = 0.0 # Setting this to 0 for our experiments
        #print(f"new home == {self.z_off_opt_candidate}")
        self.translate_eef_frame_wrt(tf_tw_init, [0, 0, self.z_off_opt_candidate])
        self.follow_path_wrt_measure_sharpness(tf_tw_init, path_init)
        print(f'initial sharpness = {self.sharpness_opt}')
        
        # if self.sharpness_opt <= threshold_sharpness:
        #     pass
        # # Set duration of trajectory execution compared to original planned time.
        # #print(f'z_off_opt = {self.z_off_opt}')
        # else:
        while self.sharpness_opt < threshold_sharpness and z_off_h < max_off and not rospy.is_shutdown():
            z_off_arr = np.linspace(self.z_off_opt_candidate + z_off_l, self.z_off_opt_candidate + z_off_h, n)
            new_start_z = np.zeros((n, 3))
            new_start_z[:, -1] = self.z_off_opt
            new_start_z[:,2] = z_off_arr

            #z_off_arr = np.linspace(z_off_l, z_off_h, n)
            path = np.zeros((n, 3))
            path[:,2] = z_off_arr
            
            
            #self.follow_path_wrt_measure_sharpness(tf_tw, path)
            #self.translate_eef_frame_wrt(tf_tw, [0, 0, z_off_h])
            self.translate_eef_frame_wrt(tf_tw_init, [0, 0, self.z_off_opt_candidate + z_off_h + self.z_off_opt])
            self.follow_path_wrt_measure_sharpness(tf_tw_init, path)
            
            #print(path)
            #print(f'while loop z_off_opt = {self.z_off_opt}')
            print(self.sharpness_opt)
            z_off_l -= 0.01
            z_off_h += 0.01
            
            #change speed scale factor based on the sharpness
            min_scaling = 0.4
            
            if self.sharpness_opt < 100:
                self.speed_scaling = 1.0
            else:
                self.speed_scaling = 0.5#((1-min_scaling)/(threshold_sharpness - 100))*(self.sharpness_opt - 100) + min_scaling
        
        if self.sharpness_opt < threshold_sharpness:
            return False
        
        if self.sharpness_opt >= threshold_sharpness:
            self.z_off_opt_candidate = self.z_off_opt
 
        #self.move_group.go(self.pose_opt, wait=True)
        (plan, fraction) = self.move_group.compute_cartesian_path([self.pose_opt], self.eef_step, self.jump_threshold, avoid_collisions=True)
        self.move_group.execute(plan, wait=True)

        #print(f"self.z_off_opt = {self.z_off_opt}")

        # Set the maximum displacement for auto-focus
        self.speed_scaling = 0.2
        #self.sharpness_opt = 0.0

        z_off_l = self.z_off_opt - 0.01 # units: meters
        z_off_h = self.z_off_opt + 0.01
        
        #print(f'after while loop z_off_opt = {self.z_off_opt}')
        #while self.sharpness_opt < threshold_sharpness and z_off_h < max_off and not rospy.is_shutdown():
            
            # Number of poses we check for optimal focus
        n = 1
        path = np.zeros((n, 3))
        # Array of offsets along z-axis of eef
        z_off_arr = np.linspace(z_off_l, z_off_h, n)
        path[:,2] = z_off_arr
        self.translate_eef_frame_wrt(tf_tw_init, [0, 0, z_off_h])
        self.follow_path_wrt_measure_sharpness(tf_tw_init, path)

        # z_off_l -= 0.001
        # z_off_h += 0.001

        # print(self.pose_opt)
        print(f'optimal sharpness = {self.sharpness_opt}')
        #print(f'optimal z_off_opt = {self.z_off_opt}')
        #print("take a photo")
        # self.move_group.go(self.pose_opt, wait=True)
        (plan, fraction) = self.move_group.compute_cartesian_path([self.pose_opt], self.eef_step, self.jump_threshold, avoid_collisions=True)
        self.move_group.execute(plan, wait=True)
        #self.move_group.set_max_velocity_scaling_factor(1) 
        
        z_off_data = []
        sharpness_data = []
        time_data = []
        i = 0
        
        with rosbag.Bag(self.bag_file_path, 'r', allow_unindexed=True) as bag:
            for topic, msg, t in bag.read_messages(topics =['/inspection/autofocus/z_off', '/inspection/autofocus/sobel', '/inspection/autofocus/time']):
                time_in_seconds = t.to_sec()
                if topic == '/inspection/autofocus/z_off':
                    z_off_data.append((time_in_seconds,msg.data))
                elif topic == '/inspection/autofocus/sobel':
                    sharpness_data.append((time_in_seconds,msg.data))      
                elif topic == '/inspection/autofocus/time':
                    time_data.append(msg.data)

        time_data = [t - time_data[0] for t in time_data]

                        

        timestamp = int(time.time())
        image_filename = f'rosbag__daself.original_joint_positionsta_z_off_vs_sharpess_img_{timestamp}_{i}_0mm_manip.PNG'
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))
        axs[0].scatter([item[1] for item in z_off_data], [item[1] for item in sharpness_data], marker='o', s=20)
        axs[0].set_xlabel('z_offset')
        axs[0].set_ylabel('Sharpness')
        axs[0].set_title('Sharpness vs. z_offset')
        axs[0].axhline(y=threshold_sharpness, color='red', linestyle='--', label='Threshold')
        axs[0].axhline(y=self.sharpness_opt, color='green', linestyle='--', label='Optimal Sharpness')
        axs[0].axvline(x=self.z_off_opt, color='blue', linestyle='--', label='Optimal Z Offset')
        axs[0].legend()
        axs[0].grid()

        axs[1].scatter(time_data, [item[1] for item in sharpness_data], marker='o', s=20)
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Sharpness')
        axs[1].set_title('Sharpness vs. Time')
        axs[1].axhline(y=threshold_sharpness, color='red', linestyle='--', label='Threshold')
        axs[1].axhline(y=self.sharpness_opt, color='green', linestyle='--', label='Optimal Sharpness')
        axs[1].legend()
        axs[1].grid()

        axs[2].scatter(time_data, [item[1] for item in z_off_data], marker='o', s=20)
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('z_offset')
        axs[2].set_title('z_offset vs. Time')
        axs[2].axhline(y=self.z_off_opt, color='blue', linestyle='--', label='Optimal Z Offset')
        axs[2].legend()
        axs[2].grid()


        plt.tight_layout()
        sub_dir = '/rosbag_data'
        local_dir = f'{str(Path.home())}/inspection/{sub_dir}'
        plt.savefig(f'{local_dir}/{image_filename}')
        i += 1
        return True

    def translate_eef_frame_wrt(self, tf_wt, trans):
        pose_tg = geometry_msgs.msg.PoseStamped()
        pose_tg.header.frame_id = 'inspection_camera_frame'
        pose_tg.pose.position.x = trans[0]
        pose_tg.pose.position.y = trans[1]
        pose_tg.pose.position.z = trans[2]
            
        goal_wg = tf2_geometry_msgs.do_transform_pose(pose_tg, tf_wt)
        goal_wg.pose.orientation = tf_wt.transform.rotation
        (plan, fraction) = self.move_group.compute_cartesian_path([goal_wg.pose], self.eef_step, self.jump_threshold, avoid_collisions=True)
        self.move_group.execute(plan, wait=True)

    def follow_path_wrt_measure_sharpness(self, tf, path):
        pose_arr = []
        for i in range(path.shape[0]):
            pose_tg = geometry_msgs.msg.PoseStamped()
            pose_tg.header.frame_id = 'inspection_camera_frame'
            pose_tg.pose.position.x = path[i,0]
            pose_tg.pose.position.y = path[i,1]
            pose_tg.pose.position.z = path[i,2]

            pose_wg = tf2_geometry_msgs.do_transform_pose(pose_tg, tf)
            pose_wg.pose.orientation = tf.transform.rotation
            pose_arr.append(pose_wg.pose)
        self.moving = True
        (plan, fraction) = self.move_group.compute_cartesian_path(pose_arr, self.eef_step, self.jump_threshold, avoid_collisions=True)
        plan = self.scale_trajectory_time(plan)
        self.move_group.execute(plan, wait=False)
        
        dt = 0.1
        while self.moving and not rospy.is_shutdown():
            t0 = time.time()
            pose = self.move_group.get_current_pose()
            tf_wt = self.tfBuffer.lookup_transform('world', 
                                            'inspection_camera_frame',
                                            rospy.Time(0),
                                            rospy.Duration(1.0))
            
            pose_t = geometry_msgs.msg.PoseStamped()
            pose_t.header.frame_id = 'inspection_camera_frame'
            # Transform pose at current inspection_camera_frame position to world frame
            pose_w = tf2_geometry_msgs.do_transform_pose(pose_t, tf_wt)
            # Transform pose of eef to autofocus origin frame
            pose_o = tf2_geometry_msgs.do_transform_pose(pose_w, self.tf_wt)
            z_off = pose_o.pose.position.z
            self.z_off_pub.publish(z_off)
            #print(f'z_off = {z_off}')

            # Add time, z_off and self.sharpness to
            t = time.time()
            #print(f't = {t}')
            datum = (t, z_off, self.sharpness)
            self.autofocus_data.append(datum)
            #for rosbag file
            self.log_autofocus_data(self.autofocus_data)

            # Check self.sharpness against self.sharpness_opt. If higher, update and record joint state
            if self.sharpness > self.sharpness_opt:
                self.sharpness_opt = self.sharpness
                self.pose_opt = pose.pose
                self.z_off_opt = z_off

            t1 = time.time()
            if t1 - t0 < dt:
                time.sleep(dt - (t1 - t0))

    def log_autofocus_data(self, autofocus_data):
        with rosbag.Bag(self.bag_file_path, 'w') as bag:
            for t, z_off_value, sharpness_value in autofocus_data:
                z_off_msg = Float64(data = z_off_value)
                sharpness_msg = Float64(data = sharpness_value)
                time_msg = Float64(data = t)
                bag.write('/inspection/autofocus/z_off', z_off_msg)
                bag.write('/inspection/autofocus/sobel', sharpness_msg)
                bag.write('/inspection/autofocus/time', time_msg)
    
    # def original_position(self):
    #     self.move_group.set_joint_value_target(self.original_joint_positions)
    #     self.move_group.go(wait=True)


    # Inspection Callback and Trigger

    def inspection_feedback_callback(self, feedback_msg):
        image_path = feedback_msg.image_path
        rospy.loginfo(feedback_msg)

        if not feedback_msg.downloaded:
            rospy.loginfo("Captured: {}".format(image_path))

            # Unpack parameters
            self.image_dict[image_path]['captured'] = True
            self.image_dict[image_path]['downloaded'] = feedback_msg.downloaded
            self.image_dict[image_path]['iso'] = feedback_msg.iso
            self.image_dict[image_path]['shutterspeed'] = feedback_msg.shutterspeed
            self.image_dict[image_path]['f_number'] = feedback_msg.f_number
            self.image_dict[image_path]['colortemperature'] = feedback_msg.colortemperature
            self.image_dict[image_path]['led_values'] = list(feedback_msg.light_control.data)
            rospy.loginfo(f'\t{self.image_dict[image_path]}')
            self.waiting = False
            
        if feedback_msg.disturbance:
            reinspect_region = self.inspection_dict[feedback_msg.name]
            rospy.loginfo(f'Re-Queuing region {reinspect_region.name} due to failed disturbances check.')
            self.inspection_queue.append(reinspect_region)

        if feedback_msg.partitioned:
            rospy.loginfo("Partitioned: {}".format(image_path))
            self.image_dict[image_path]['downloaded'] = feedback_msg.downloaded
            self.image_dict[image_path]['partitioned_dir_path'] = feedback_msg.partitioned_dir_path
            rospy.loginfo(f'\t{self.image_dict[image_path]}')

        if feedback_msg.classified:
            rospy.loginfo("Classified: {}".format(image_path))
            self.image_dict[image_path]['classification'] = list(feedback_msg.classification)
            rospy.loginfo(f'\t{self.image_dict[image_path]}')

    def trigger(self, name):
        time.sleep(0.5)

        sub_dir = Path('Training')
        local_dir = str(Path.home() / 'inspection' / sub_dir)

        if not os.path.exists(local_dir):
            rospy.loginfo(f'Creating directory {local_dir}...')
            os.makedirs(local_dir)
            rospy.loginfo(f'Created directory {local_dir}!')
        
        mmddyyyyhhmmss = time.strftime('%m%d%Y%H%M%S')
        part_number = 'PartNumber'
        image_name = f'{mmddyyyyhhmmss}_{part_number}_{name}.jpeg'

        image_path = str(sub_dir / image_name)

        self.image_dict[image_path] = {
            'captured': False,
            'classification': []
        }

        rospy.loginfo("Triggering!")
        req = InspectionTriggerRequest()
        req.name = name
        req.image_path = image_path
        req.training = True
        req.priority = False
        req.model = '30'
        self.inspection_trigger(req)

    # -----------------------------------------------------
    # CALIBRATION

    def calibration_callback(self, req):
        rospy.loginfo("Hello world!")
        self.init_autofocus_calibration()
        pose = Pose()
        pose.position.z = self.z_off_opt
        return pose

    def init_autofocus_calibration(self,z_off_init=0.0):
        #pose_tg = geometry_msgs.msg.PoseStamped()
        tf_tw_init = self.tfBuffer.lookup_transform('world', 
                                                'inspection_camera_frame',
                                                rospy.Time(0),
                                                rospy.Duration(1.0))

        # transform: world to inspection_camera_frame, used for logging offset data
        self.tf_wt = self.tfBuffer.lookup_transform('inspection_camera_frame', 
                                                'world',
                                                rospy.Time(0),
                                                rospy.Duration(1.0))
        # Set the baseline variables for checking focus
        self.speed_scaling = 0.1
        self.sharpness_opt = 0.1
        self.pose_opt = None
        self.z_off_opt = 0
        self.z_off_opt_candidate = 0
        z_off_l = -0.0 # units: meters
        z_off_h = 0.2
        n = 2
                
        
        
        path = np.zeros((n, 3))
        z_off_arr_l = np.linspace(z_off_l, 0, n)
        path[:,2] = z_off_arr_l
        
        self.translate_eef_frame_wrt(tf_tw_init, [0, 0, 0])
        self.follow_path_wrt_measure_sharpness(tf_tw_init, path)
        
        if self.z_off_opt < 300:
            path = np.zeros((n, 3))
            z_off_arr_h = np.linspace(0, z_off_h, n)
            path[:,2] = z_off_arr_h
            self.translate_eef_frame_wrt(tf_tw_init, [0, 0, 0])
            self.follow_path_wrt_measure_sharpness(tf_tw_init, path)
            (plan, fraction) = self.move_group.compute_cartesian_path([self.pose_opt], self.eef_step, self.jump_threshold, avoid_collisions=True)
            self.move_group.execute(plan, wait=True)
        else:
            (plan, fraction) = self.move_group.compute_cartesian_path([self.pose_opt], self.eef_step, self.jump_threshold, avoid_collisions=True)
            self.move_group.execute(plan, wait=True)
            
        # print(self.pose_opt)
        print(self.sharpness_opt)
        
        return True    


    def calibrate(self):
        n = 0
        while True:
            k = input('Ready for image?')
            if k == 'x': return
            image_path = f'/Training/calib_img_{n}.jpeg'
            sub_dir = Path('Calibration')
            local_dir = str(Path.home() / 'inspection' / sub_dir)
            
            if not os.path.exists(local_dir):
                rospy.loginfo(f'Creating directory {local_dir}...')
                os.makedirs(local_dir)
                rospy.loginfo(f'Created directory {local_dir}!')
            
            mmddyyyyhhmmss = time.strftime('%m%d%Y%H%M%S')
            part_number = 'Calibration'
            image_name = f'{mmddyyyyhhmmss}_{part_number}.jpeg'

            image_path = str(sub_dir / image_name)
            
            rospy.loginfo("Triggering!")
            req = InspectionTriggerRequest()
            req.image_path = image_path
            req.training = True
            req.priority = False
            req.model = '30'
            self.inspection_trigger(req)
            n += 1


if __name__=="__main__":
    client = InspectionRobotService()
    client.run()
    #client.calibrate()
