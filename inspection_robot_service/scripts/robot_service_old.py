#!/usr/bin/env python3
import sys, os, time
import numpy as np
import copy
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, curve_fit
import csv
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

# Service Imports
from inspection_vision.srv import GetSharpness
from inspection_task_planning.srv import InspectionTrigger, InspectionTriggerRequest

# Action Imports
from inspection_task_planning.msg import InspectionAction, InspectionGoal

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
        # INSPECTION POSES

        self.state = WAITING
        self.inspection_poses = None # List of poses in 'world' coordinate frame we will visit during inspection.

        # TODO: Subscribe to /inspection/auto_partitioning/poses and implement auto_partitioning_callback() function to
        #       save poses in world coordinate frame.
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
        self.bag_file_path = '/home/arun/inspection/rosbag_data/autofocus_data.bag'

        # -----------------------------------------------------
        # ROBOT SERVICE

        self.bridge = CvBridge()

        self.sharpness = 0
        self.sharpness_sub = rospy.Subscriber('/inspection/perception/sobel', Float64, self.sharpness_callback)

        self.roi = None
        self.roi_sub = rospy.Subscriber('/inspection/camera/focus/image', Image, self.roi_callback)

        self.pose_opt = None
        self.z_off_opt = 0
        self.z_off_opt_candidate = 0
        self.sharpness_opt = 0
        self.joint_fb_sub = rospy.Subscriber('/scaled_pos_joint_traj_controller/follow_joint_trajectory/feedback', FollowJointTrajectoryActionFeedback, self.joint_fb_callback)
        
        # For measuring sharpness
        self.moving = False
        self.ex_traj_result_sub = rospy.Subscriber('/execute_trajectory/result', ExecuteTrajectoryActionResult, self.ex_traj_result_callback)

        self.offset = 0
        self.tf_wt = self.tfBuffer.lookup_transform('world', 
                                                    'inspection_camera_frame',
                                                    rospy.Time(0),
                                                    rospy.Duration(1.0))
        
        
        # self.step_server = rospy.Service('/inspection/robot/step', Step, self.step_callback)
        # self.reset_server = rospy.Service('/inspection/robot/reset', Reset, self.reset_callback)

        
        
    # -----------------------------------------------------
    # ROBOT SERVICE

    def joint_fb_callback(self, msg):
        pose = self.move_group.get_current_pose()
        sharpness = self.sharpness
        
        tf_wt = self.tfBuffer.lookup_transform('world', 
                                                'inspection_camera_frame',
                                                rospy.Time(0),
                                                rospy.Duration(1.0))
        pose_t = geometry_msgs.msg.PoseStamped()
        pose_t.header.frame_id = 'inspection_camera_frame'
        # Transform pose at current tool0 position to world frame
        pose_w = tf2_geometry_msgs.do_transform_pose(pose_t, tf_wt)
        # Transform pose of eef to autofocus origin frame
        pose_o = tf2_geometry_msgs.do_transform_pose(pose_w, self.tf_wt)
        z_off = pose_o.pose.position.z
        self.z_off_pub.publish(z_off)
                
        if sharpness > self.sharpness_opt:
            self.sharpness_opt = sharpness
            self.pose_opt = pose.pose

    def ex_traj_result_callback(self, msg):
        #print("Hello world.")
        self.moving = False

    def sharpness_callback(self, msg):
        self.sharpness = msg.data

    def roi_callback(self, msg):
        self.roi = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def run(self):
        while not rospy.is_shutdown():
            # Wait for list of imaging poses
            while self.state == WAITING and not rospy.is_shutdown():
                rospy.Rate(10).sleep

            # Execute inspection regime
            self.inspect_part()
       
    def quit(self):
        rospy.loginfo('Cancelling inspection goal...')
        self.inspection_client.cancel_all_goals()

    # -----------------------------------------------------
    # INSPECTION POSES

    def auto_partitioning_callback(self, msg):
        # TODO: msg received is a PoseArray. All poses in the list are in the part coordinate frame (string value stored in msg.header.frame_id).
        #       Iterate through the Pose messages in msg.poses, turning them into PoseStamped with the same header as the PoseArray and save them into
        #       self.inspection_poses list.

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

        # Example of how to transform a PoseStamped with frame_id = msg.header.frame_id to frame_id = 'world'.
        # ps_p = PoseStamped()
        # ps_p.header = msg.header
        # ps_p.pose = msg.poses[0]
        # ps_w = tf2_geometry_msgs.do_transform_pose(ps_p, tf_part_to_world)

        self.state = READY


    # -----------------------------------------------------
    # INSPECTION REGIMES


    def inspect_part(self):
        # TODO: Move to poses saved in self.inspection_poses. Between poses, add the following line so we don't immediately move on to the next pose. 
        # keyboard = Controller()
        for i in range(len(self.inspection_poses)):
            # ready_pose = input('Ready for next pose?')
            # if ready_pose == "":
            # Captures situations where the input is just ENTER key
            pose = self.inspection_poses[i].pose
            # if i == 0:
            #     pose = self.inspection_poses[i].pose
            #     print(pose.position.z)
            # else:
            #     pose.position.z = self.z_off_opt
            #     print(pose.position.z)
            # (plan, fraction) = self.move_group.compute_cartesian_path([pose], 0.001, 0.0) # pose is not stamped. Assumed to be in 'world' frame.
            self.move_group.set_max_velocity_scaling_factor(0.5) 
            self.move_group.set_pose_target(pose)
            self.move_group.go(wait=True)
            self.inspect(f'image_{i}')
            # print(fraction)
            # input('check plan')
            # If fraction < 1, planning to pose failed, most likely because it is impossible to move there.
            # if(fraction==1):
                # time.sleep(10)
                # self.move_group.execute(plan, wait=True)  
                # self.inspect(f'image_{i}')
            # else:   
                # continue     
        
        # In this case, skip this pose.

        # Wait for next list of inspection poses.
        self.state = WAITING

    def image_acrylic_coupon(self):
        # TF world to focus plane
        tf_b_to_fp = self.tfBuffer.lookup_transform('world', 
                                        'focus_plane',
                                        rospy.Time(0),
                                        rospy.Duration(1.0))
        # TF world to coupon_surface
        tf_b_to_cs = self.tfBuffer.lookup_transform('world', 
                                        'coupon_surface',
                                        rospy.Time(0),
                                        rospy.Duration(1.0))

        print(tf_b_to_fp.transform.rotation)
        print(tf_b_to_cs.transform.translation)

        xy_origin = tf_b_to_cs.transform.translation
        down = tf_b_to_fp.transform.rotation

        origin = Pose()
        origin.position = xy_origin
        origin.position.z = origin.position.z + 0.39375
        origin.orientation = down
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose = origin
        self.pose_pub.publish(pose_msg)

        # (plan, fraction) = self.move_group.compute_cartesian_path([origin], 0.001, 0.0)
        # self.move_group.execute(plan, wait=True)

        img_pose = Pose()
        img_pose.position.x = origin.position.x
        img_pose.position.y = origin.position.y
        img_pose.position.z = origin.position.z
        img_pose.orientation = origin.orientation
        for pos in [(4,5)]:
            i = pos[0]
            j = pos[1]
            img_pose.position.x = origin.position.x + i*0.075
            img_pose.position.y = origin.position.y + j*0.1
            print(origin)

            (plan, fraction) = self.move_group.compute_cartesian_path([img_pose], 0.001, 0.0)
            self.move_group.execute(plan, wait=True)

            input('Ready?')

            # self.inspect(f'i{i}j{j}')


    # -----------------------------------------------------
    # ROBOT MOVEMENT

    def get_joint_state(self):
        print(self.move_group.get_current_state())

    def return_home(self):
        success = self.move_group.go(self.home_joints, wait=True)
        self.move_group.stop()
        return success

    
    def autofocus(self, z_off=None):
        self.autofocus_data = []
        # Look up the transform to current location of eef
        # transform: tool0 to world
        self.move_group.set_max_velocity_scaling_factor(1) 
        tf_tw = self.tfBuffer.lookup_transform('world', 
                                                'inspection_camera_frame',
                                                rospy.Time(0),
                                                rospy.Duration(1.0))
        # transform: world to tool0, used for logging offset data
        self.tf_wt = self.tfBuffer.lookup_transform('inspection_camera_frame', 
                                                'world',
                                                rospy.Time(0),
                                                rospy.Duration(1.0))
        # Set the baseline variables for checking focus
        self.sharpness_opt = 0
        self.pose_opt = None
        #self.z_off_opt = 0

        # Set the maximum displacement for auto-focus
        z_off_l = -0.01 # units: meters
        z_off_h = 0.01
        n = 1
        max_off = 0.1
        threshold_sharpness = 100
        print(f"new home == {self.z_off_opt_candidate}")
                
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
            self.follow_path_wrt_measure_sharpness(tf_tw, path)
            self.translate_eef_frame_wrt(tf_tw, [0, 0, self.z_off_opt_candidate + z_off_h])
            
            #print(path)
            print(self.sharpness_opt)
            z_off_l -= 0.01
            z_off_h += 0.01
        
        if self.sharpness_opt < threshold_sharpness:
            return False
        
        if self.sharpness_opt >= threshold_sharpness:
            self.z_off_opt_candidate = self.z_off_opt
 
        #self.move_group.go(self.pose_opt, wait=True)
        (plan, fraction) = self.move_group.compute_cartesian_path([self.pose_opt], self.eef_step, self.jump_threshold, avoid_collisions=True)
        self.move_group.execute(plan, wait=True)

        #print(f"self.z_off_opt = {self.z_off_opt}")

        # Set the maximum displacement for auto-focus
        z_off_l = self.z_off_opt - 0.007 # units: meters
        z_off_h = self.z_off_opt + 0.007
        # Number of poses we check for optimal focus
        n = 1
        path = np.zeros((n, 3))
        # Array of offsets along z-axis of eef
        z_off_arr = np.linspace(z_off_l, z_off_h, n)
        path[:,2] = z_off_arr
        self.follow_path_wrt_measure_sharpness(tf_tw, path)
        self.translate_eef_frame_wrt(tf_tw, [0, 0, z_off_h])

        # print(self.pose_opt)
        print(self.sharpness_opt)
        #print("take a photo")
        # self.move_group.go(self.pose_opt, wait=True)
        (plan, fraction) = self.move_group.compute_cartesian_path([self.pose_opt], self.eef_step, self.jump_threshold, avoid_collisions=True)
        self.move_group.execute(plan, wait=True)
        #self.move_group.set_max_velocity_scaling_factor(1) 
        
        return True

        # z_off_data = []
        # sharpness_data = []

        # # Save autofocus_data to bag file and read it and plotting a graph
        # with rosbag.Bag(self.bag_file_path, 'r', allow_unindexed=True) as bag:
        #     for topic, msg, t in bag.read_messages(topics =['/inspection/autofocus/z_off', '/inspection/autofocus/sobel']):
        #         if topic == '/inspection/autofocus/z_off':
        #             z_off_data.append(msg.data)
        #         elif topic == '/inspection/autofocus/sobel':
        #             sharpness_data.append(msg.data)                

        # plt.figure(figsize=(10, 6))
        # plt.scatter(z_off_data, sharpness_data, marker='o', s=20)
        # plt.xlabel('z_offset')
        # plt.ylabel('Sharpness')
        # plt.title('z_offset vs Sharpness')
        # plt.grid()
        # plt.show()

    # def translate_eef_frame_wrt(self, tf_wt, trans):
    #     pose_tg = geometry_msgs.msg.PoseStamped()
    #     pose_tg.header.frame_id = 'inspection_camera_frame'
    #     pose_tg.pose.position.x = trans[0]
    #     pose_tg.pose.position.y = trans[1]
    #     pose_tg.pose.position.z = trans[2]
    #     goal_wg = tf2_geometry_msgs.do_transform_pose(pose_tg, tf_wt)
    #     goal_wg.pose.orientation = tf_wt.transform.rotation
    #     # self.move_group.go(goal_wg.pose, wait=True)
    #     (plan, fraction) = self.move_group.compute_cartesian_path([self.pose_opt], self.eef_step, self.jump_threshold, avoid_collisions=True)
    #     self.move_group.execute(plan, wait=True)

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

    # def follow_path_wrt(self, tf_wt, path):
    #     pose_arr = []
    #     for i in range(path.shape[0]):
    #         pose_tg = geometry_msgs.msg.PoseStamped()
    #         pose_tg.header.frame_id = 'tool0'
    #         pose_tg.pose.position.x = path[i,0]
    #         pose_tg.pose.position.y = path[i,1]
    #         pose_tg.pose.position.z = path[i,2]
    #         pose_wg = tf2_geometry_msgs.do_transform_pose(pose_tg, tf_wt)
    #         pose_wg.pose.orientation = tf_wt.transform.rotation
    #         pose_arr.append(pose_wg.pose)

    #     (plan, fraction) = self.move_group.compute_cartesian_path(pose_arr, self.eef_step, self.jump_threshold, avoid_collisions=True)
    #     self.move_group.execute(plan, wait=True)

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
            # Transform pose at current tool0 position to world frame
            pose_w = tf2_geometry_msgs.do_transform_pose(pose_t, tf_wt)
            # Transform pose of eef to autofocus origin frame
            pose_o = tf2_geometry_msgs.do_transform_pose(pose_w, self.tf_wt)
            z_off = pose_o.pose.position.z
            self.z_off_pub.publish(z_off)

            # Add time, z_off and self.sharpness to
            t = time.time()
            datum = (t, z_off, self.sharpness)
            self.autofocus_data.append(datum)
            #for rosbag file
            #self.log_autofocus_data(self.autofocus_data)

            # Check self.sharpness against self.sharpness_opt. If higher, update and record joint state
            if self.sharpness > self.sharpness_opt:
                self.sharpness_opt = self.sharpness
                self.pose_opt = pose.pose
                self.z_off_opt = z_off

            t1 = time.time()
            if t1 - t0 < dt:
                time.sleep(dt - (t1 - t0))

        self.autofocus_data.extend(self.autofocus_data)
            #print(self.moving)
        #print(len(self.autofocus_data))

    def log_autofocus_data(self, autofocus_data):
        bag_dir = '/home/arun/inspection/rosbag_data'
        bag_file_path = os.path.join(bag_dir, 'autofocus_data.bag')

        with rosbag.Bag(self.bag_file_path, 'w') as bag:
            for t, z_off_value, sharpness_value in autofocus_data:
                z_off_msg = Float64(data = z_off_value)
                sharpness_msg = Float64(data = sharpness_value)
                bag.write('/inspection/autofocus/z_off', z_off_msg)
                bag.write('/inspection/autofocus/sobel', sharpness_msg)




    # Inspection Callback and Trigger

    def inspect(self, img_name):
        #self.autofocus()
        focused = True
        # focused = self.autofocus()
        if focused:
            self.trigger(img_name)

    def inspection_feedback_callback(self, feedback_msg):
        image_path = feedback_msg.image_path

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

        if feedback_msg.partitioned:
            rospy.loginfo("Partitioned: {}".format(image_path))
            self.image_dict[image_path]['downloaded'] = feedback_msg.downloaded
            self.image_dict[image_path]['partitioned_dir_path'] = feedback_msg.partitioned_dir_path
            rospy.loginfo(f'\t{self.image_dict[image_path]}')

        if feedback_msg.classified:
            rospy.loginfo("Classified: {}".format(image_path))
            self.image_dict[image_path]['classification'] = list(feedback_msg.classification)
            rospy.loginfo(f'\t{self.image_dict[image_path]}')

    def trigger(self, img_name):
        time.sleep(0.5)

        sub_dir = '/Training'

        local_dir = f'{str(Path.home())}/inspection/{sub_dir}'

        if not os.path.exists(local_dir):
            rospy.loginfo(f'Creating directory {local_dir}...')
            os.makedirs(local_dir)
            rospy.loginfo(f'Created directory {local_dir}!')
        
        mmddyyyyhhmmss = time.strftime('%m%d%Y%H%M%S')
        part_number = 'PartNumber'
        region_name = 'RegionName'
        image_name = f'{mmddyyyyhhmmss}_{part_number}_{region_name}'

        image_path = f'{sub_dir}/{image_name}.jpeg'

        self.image_dict[image_path] = {
            'captured': False,
            'classification': []
        }

        rospy.loginfo("Triggering!")
        req = InspectionTriggerRequest()
        req.image_path = image_path
        req.training = True
        req.priority = False
        req.model = '30'
        self.inspection_trigger(req)

    def calibrate(self):
        n = 0
        while True:
            k = input('Ready for image?')
            if k == 'x': return
            image_path = f'/Training/calib_img_{n}.jpeg'
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
