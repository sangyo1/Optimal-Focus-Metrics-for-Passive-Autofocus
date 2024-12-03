#!/usr/bin/env python3
import time
import math
import numpy as np
import pytransform3d.rotations as pr
from dataclasses import dataclass

# ROS Imports
import rospy
import rospkg

from geometry_msgs.msg import Pose, PoseArray
import sensor_msgs.msg



@dataclass
class Calibration():
    board_length: float=180
    board_width: float=1200
    tilt_angle: float=math.radians(15)
    rotate_angle: float=math.radians(90)
    offset: int=400
    
    def rot_about_z(self,angle):
        xhat=np.asarray([-math.cos(math.radians(angle)),-math.sin(math.radians(angle)),0])
        yhat=np.asarray([-math.sin(math.radians(angle)),math.cos(math.radians(angle)),0])
        return xhat,yhat
    
    def get_pose(Normals,Point):
        tvec = Point/1000
        quat =pr.quaternion_xyzw_from_wxyz(pr.quaternion_from_matrix(Normals))
        P = [quat,tvec]
        return  P

    def get_calib_waypoints(self):
        poses=[]
        orgin_point=np.array([0, 0, 0])
        orgin_normal=np.array([0, 0, 1])
        offset_point=np.array([0, 0, self.offset])
        home= orgin_point+offset_point
        waypoint_xyz=[]
        waypoint_N=[]
        home_normal=np.asarray([[1, 0, 0],[0,-1,0],[0,0,-1]])
        waypoint_N.append(home_normal)
        waypoint_xyz.append(home) # 1 pose
        waypoint_generation=home
        
        #waypoints placing the board on the extremes in all axes, 6 poses
        for i in range(3):
            waypoint_generation[i]+=self.board_width/4
            waypoint_xyz.append(waypoint_generation.copy())  # Use a copy to avoid modifying the same instance
            waypoint_N.append(home_normal)
            waypoint_generation[i]-=self.board_width/2  # Subtract half of the width instead of 1/4
            waypoint_xyz.append(waypoint_generation.copy())  # Use a copy to avoid modifying the same instance
            waypoint_N.append(home_normal)
            waypoint_generation[i]+=self.board_width/4  # Return to the original position for the next axis
        
  
        #waypoint rotation about z-axis by 15 degree, 2 poses
        for x in range(2):
            x_hat,y_hat=self.rot_about_z(((-1)**(x+1))*15)
            z_hat=np.asarray([0,0,-1])            
            waypoint_xyz.append(home.copy())
            waypoint_N.append(np.hstack((x_hat.reshape(3,1), y_hat.reshape(3,1), z_hat.reshape(3,1))))

        #waypoint rotation about z-axis by 90 degree, 1 pose
        x_hat,y_hat=self.rot_about_z(90)
        z_hat=np.asarray([0,0,-1])            
        waypoint_xyz.append(home.copy())
        waypoint_N.append(np.hstack((x_hat.reshape(3,1), y_hat.reshape(3,1), z_hat.reshape(3,1))))
        
 
        # waypoints for placing the board with tilt wrt x and y, 4 poses
        # for j in range(2):
        #     waypoint_generation[j]=-self.offset*math.sin(self.tilt_angle)
        #     waypoint_generation[2]=self.offset*math.cos(self.tilt_angle)
        #     z_hat = (orgin_point-waypoint_generation.copy())/np.linalg.norm(waypoint_generation.copy()-orgin_point)
        #     y_hat = -np.cross(z_hat, orgin_normal)
        #     y_hat=y_hat/np.linalg.norm(y_hat)
        #     x_hat = np.cross(z_hat, y_hat)
        #     x_hat=x_hat/np.linalg.norm(x_hat)
        #     waypoint_N.append(np.hstack((x_hat.reshape(3,1), y_hat.reshape(3,1), z_hat.reshape(3,1))))
        #     waypoint_xyz.append(waypoint_generation.copy())
        #     waypoint_generation=home
        #     waypoint_generation[j]=self.offset*math.sin(self.tilt_angle)
        #     waypoint_generation[2]=self.offset*math.cos(self.tilt_angle)
        #     z_hat = (orgin_point-waypoint_generation.copy())/np.linalg.norm(waypoint_generation.copy()-orgin_point)
        #     y_hat = np.cross(z_hat, orgin_normal)
        #     y_hat=y_hat/np.linalg.norm(y_hat)
        #     x_hat = np.cross(z_hat, y_hat)
        #     x_hat=x_hat/np.linalg.norm(x_hat)
        #     waypoint_N.append(np.hstack((x_hat.reshape(3,1), y_hat.reshape(3,1), z_hat.reshape(3,1))))
        #     waypoint_xyz.append(waypoint_generation.copy())
        #     waypoint_generation=home

        # print(waypoint_xyz)
        for k in range(len(waypoint_xyz)):
            # print(waypoint_N[k])
            # print(k)
            
            tvec = waypoint_xyz[k]/1000
            quat =pr.quaternion_xyzw_from_wxyz(pr.quaternion_from_matrix(waypoint_N[k]))
            pose = [quat,tvec]
            poses.append(pose)
            
        return poses
    
# Initialize ROS node
rospy.init_node('calib_vp_node')

rospy.loginfo("Starting calib_vp_node...")

# Create PoseArray Publisher. Publish to topic '/inspection/auto_partitioning/poses'

posearray_publisher = rospy.Publisher('/inspection/auto_partitioning/poses', PoseArray)
    
calib_vp=Calibration()
pose_arr=PoseArray()
pose_list = calib_vp.get_calib_waypoints()
print(len(pose_list))
poses=[]
for i in range(len(pose_list)):
    pose= Pose()
    pose.position.x, pose.position.y, pose.position.z = pose_list[i][1]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = pose_list[i][0]
    poses.append(pose)

# print(poses)        
pose_arr.header.frame_id = 'fixture'
pose_arr.poses=poses
print(pose_arr)

time.sleep(2)

rospy.loginfo('Publishing PoseArray to /inspection/auto_partitioning/poses')

posearray_publisher.publish(pose_arr)
