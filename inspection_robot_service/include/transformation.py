# collection of helper functions for spatial transformations
# created by Hui Xiao @ University of Washington 2020
# dependency: pytransform3d. See https://github.com/rock-learning/pytransform3d
import numpy as np
from geometry_msgs.msg import TransformStamped, Transform, Pose, PoseStamped
import pytransform3d.rotations as pr
import rospy

class Trans3D:
    def __init__(self, translation, quaternion):
        self.tvec = translation.flatten().astype('float64')
        # quaternion [x,y,z,w]
        self.quat = quaternion.flatten().astype('float64')

    @staticmethod
    def from_tfmatrix(tfmatrix):
        '''
        Created transformation from 4x4 homogeneous transformation matrix
        '''
        tvec = tfmatrix[0:3,3]     
        rot = tfmatrix[0:3, 0:3]
        quat = pr.quaternion_xyzw_from_wxyz(pr.quaternion_from_matrix(rot))
        return Trans3D(tvec, quat)
    
    def to_tfmatrix(self):
        '''
        return 4x4 homogeneous transformation matrix
        '''
        out = np.eye(4)
        rot = self.to_rotmatrix()
        out[0:3, 0:3] = rot
        out[0:3, 3] = self.tvec
        return out

    @staticmethod
    def from_quaternion(quat, tvec = np.array([0,0,0])):
        '''
        Create transformation from quaternion [x,y,z,w]
        '''
        return Trans3D(tvec, quat)
    
    def to_quaternion(self):
        return self.quat.copy()

    @staticmethod
    def from_rotmatrix(rotmatrix, tvec = np.array([0,0,0])): 
        '''
        Create transformation from 3x3 rotation matrix, assuming zero translation
        '''
        quat = pr.quaternion_xyzw_from_wxyz(pr.quaternion_from_matrix(rotmatrix))
        return Trans3D(tvec, quat)
    
    def to_rotmatrix(self):
        '''
        return 3x3 rotation matrix
        '''
        return pr.matrix_from_quaternion(pr.quaternion_wxyz_from_xyzw(self.quat))

    @staticmethod
    def from_angaxis(angaxis, tvec = np.array([0,0,0])):
        '''
        Create transformation from angle axis (array of size 3).
        '''
        angaxis = angaxis.flatten()
        quat = pr.quaternion_from_axis_angle(pr.axis_angle_from_compact_axis_angle(angaxis))
        return Trans3D(tvec, pr.quaternion_xyzw_from_wxyz(quat))

    def to_angaxis(self):
        '''
        return angle axis
        '''
        quat = pr.quaternion_wxyz_from_xyzw(self.quat)
        angaxis = pr.compact_axis_angle(pr.axis_angle_from_quaternion(quat))
        return angaxis

    @staticmethod
    def from_tvec(tvec):
        '''
        Create transformation from translation vector, assuming no rotation 
        '''
        tvec = tvec.flatten()
        return Trans3D(tvec, np.array([0,0,0,1]))
    
    def to_tvec(self):
        '''
        return 3x1 translation vector
        '''
        return self.tvec.reshape((3,1)).copy()
    
    @staticmethod
    def from_Pose(msg):
        '''
        Create transformation from Pose ROS message
        '''
        t = msg.position
        r = msg.orientation
        return Trans3D(np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w]))
    
    def to_Pose(self):
        '''
        return pose as Pose ROS message
        '''
        out = Pose()
        out.position.x = self.tvec[0]
        out.position.y = self.tvec[1]
        out.position.z = self.tvec[2]
        out.orientation.x = self.quat[0]
        out.orientation.y = self.quat[1]
        out.orientation.z = self.quat[2]
        out.orientation.w = self.quat[3]
        return out

    @staticmethod
    def from_PoseStamped(msg):
        '''
        Create transformation from PoseStamped ROS message
        '''
        return Trans3D.from_Pose(msg.pose)

    def to_PoseStamped(self):
        '''
        Return pose as PoseStamped ROS message
        '''
        out = PoseStamped()
        out.pose = self.to_Pose()
        return out
        
    @staticmethod
    def from_Transform(msg):
        '''
        Create transformation from Transform ROS message
        '''
        t = msg.translation
        r = msg.rotation
        return Trans3D(np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w]))

    def to_Transform(self):
        '''
        Return pose as Transform ROS message
        '''
        out = Transform()
        out.translation.x = self.tvec[0]
        out.translation.y = self.tvec[1]
        out.translation.z = self.tvec[2]
        out.rotation.x = self.quat[0]
        out.rotation.y = self.quat[1]
        out.rotation.z = self.quat[2]
        out.rotation.w = self.quat[3]
        return out

    @staticmethod
    def from_TransformStamped(msg):
        return Trans3D.from_Transform(msg.transform)

    def to_TransformStamped(self):
        out = TransformStamped()
        out.transform = self.to_Transform()
        return out
    
    @staticmethod
    def from_ROSParameterServer(param_name):
        '''
        Get pose from ros parameter server. The parameter must available in ROS server with structure:
            param_name:
                translation: [x, y, z] 
                rotation: [i, j, k, w] # rotation as quternion
        '''
        trans = rospy.get_param(param_name)
        rot = np.array(trans["rotation"])
        tvec = np.array(trans["translation"])
        pose = Trans3D.from_quaternion(rot, tvec)
        return pose
    
    @staticmethod
    def from_dict(dict):
        '''
        Get pose from python dictionary
        dict = {'translation': [x, y, z], 'rotation': [qx, qy, qz, qw]}
        '''
        rot = np.array(dict["rotation"])
        tvec = np.array(dict["translation"])
        pose = Trans3D.from_quaternion(rot, tvec)
        return pose
    
    def to_yamlString(self, pose_name, indent=0):
        ident_str = "  " * indent
        string = ident_str + pose_name + ":\n"
        string += ident_str + "  translation: [{}, {}, {}]\n".format(self.tvec[0], self.tvec[1], self.tvec[2])   
        string += ident_str + "  rotation: [{}, {}, {}, {}]\n".format(self.quat[0], self.quat[1], self.quat[2], self.quat[3])   
        return string

    def to_string(self):
        return "rotation: {},  translation: {}".format(self.quat, self.tvec)
    
    def __str__(self):
        return self.to_string()
    
    def __mul__(self, other):
        m1 = self.to_tfmatrix()
        m2 = other.to_tfmatrix()
        m = np.matmul(m1, m2)
        return Trans3D.from_tfmatrix(m)
        
if __name__ == "__main__":
    rot = np.ones((3,3))
    vec = np.ones((3,1))
    pose1 = Trans3D.from_tfmatrix(np.eye(4))
    print(pose1.to_string())
    pose2 = Trans3D.from_rotmatrix(np.eye(3))
    print(pose2.to_string())
    pose3 = Trans3D.from_tvec(np.array([1,2,3]))
    print(pose3.to_string())
    msg = pose3.to_Pose()
    print(msg)
    pose3 = Trans3D.from_Pose(msg)
    print(pose3.to_string())
    msg = pose3.to_TransformStamped()
    print(msg)
    pose3 = Trans3D.from_TransformStamped(msg)
    print(pose3.to_string())
    pose4 = Trans3D.from_angaxis(np.array([1,1,1]))
    print(pose4.to_string())
    print(pose4.to_angaxis())
    print(pose4.to_tfmatrix())
    print(pose4.to_yamlString("pose", 1))
    print("testing printing")
    print(pose1)
    print("testing * operation")
    pose1 = Trans3D.from_tvec(np.array([1,2,3]))
    pose2 = Trans3D.from_tvec(np.array([4,5,6]))
    print(pose1 * pose2)