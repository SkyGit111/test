import sys
import threading
import time
import numpy as np
from time import sleep
from collections import deque
from scipy.signal import butter, filtfilt
import rospy
import tf.transformations as tf_trans
import geometry_msgs.msg
import moveit_commander


vp_pose_list_right = deque(maxlen=10)  
vp_pose_list_left  = deque(maxlen=10)  
latest_vp_pose_right = None
latest_vp_pose_left = None
vp_pub_right = None
vp_pub_left  = None
data_lock = threading.Lock()

def butterworth_lowpass_filter(data, cutoff=2, fs=10, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data, axis=0)
    return filtered

def filter_multiple_frames(vp_pose_list, cutoff=2, fs=10, order=2):
    positions = np.array([pose[0:3, 3] for pose in vp_pose_list])
    positions_filtered = butterworth_lowpass_filter(positions, cutoff=cutoff, fs=fs, order=order)
    filtered_list = []
    for idx, pose in enumerate(vp_pose_list):
        new_pose = pose.copy()
        new_pose[0:3, 3] = positions_filtered[idx]
        filtered_list.append(new_pose)
    return np.array(filtered_list)

def transform_pose(pose_np, T):
    return np.dot(T, pose_np)

def numpy_to_pose_stamped(pose_np, frame_id="base"):
    pose_stamped = geometry_msgs.msg.PoseStamped()
    pose_stamped.header.frame_id = frame_id
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = pose_np[0, 3]
    pose_stamped.pose.position.y = pose_np[1, 3]
    pose_stamped.pose.position.z = pose_np[2, 3]
    quat = tf_trans.quaternion_from_matrix(pose_np)
    pose_stamped.pose.orientation.x = quat[0]
    pose_stamped.pose.orientation.y = quat[1]
    pose_stamped.pose.orientation.z = quat[2]
    pose_stamped.pose.orientation.w = quat[3]
    return pose_stamped

def map_operational_space(vp_pose):
    target_pose = vp_pose.copy()
    return target_pose

def init_moveit():
    """
    初始化 MoveIt 接口，返回 RobotCommander, PlanningSceneInterface,
    以及分别对应右臂和左臂的 MoveGroupCommander 对象。
    """
    moveit_commander.roscpp_initialize(sys.argv)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    # 检查 MoveIt 配置的规划组名称
    available_groups = robot.get_group_names()
    if "right_arm" not in available_groups or "left_arm" not in available_groups:
        raise ValueError(f"MoveIt 配置错误！可用的规划组: {available_groups}")

    group_right = moveit_commander.MoveGroupCommander("right_arm")
    group_left  = moveit_commander.MoveGroupCommander("left_arm")

    group_right.set_planning_time(5)
    group_right.set_goal_joint_tolerance(0.01)
    group_left.set_planning_time(5)
    group_left.set_goal_joint_tolerance(0.01)

    return robot, scene, group_right, group_left

def plan_and_execute_pose(group, target_pose_msg):
    group.set_pose_target(target_pose_msg)
    rospy.loginfo("开始规划运动轨迹...")
    plan = group.plan()
    if plan and len(plan.joint_trajectory.points) > 0:
        rospy.loginfo("规划成功，执行运动...")
        group.go(wait=True)
    else:
        rospy.logwarn("规划失败，无法执行运动。")

def vision_pro_data_stream():
    from avp_stream import VisionProStreamer
    avp_ip = "192.168.1.102"  # 根据实际情况修改
    s = VisionProStreamer(ip=avp_ip, record=True)
    rate = rospy.Rate(10)  
    global vp_pose_list_right, vp_pose_list_left, latest_vp_pose_right, latest_vp_pose_left, vp_pub_right, vp_pub_left
    while not rospy.is_shutdown():
        r = s.latest
        vp_pose_right = r['right_wrist'][0]
        vp_pose_left  = r['left_wrist'][0]
        with data_lock:
            latest_vp_pose_right = vp_pose_right
            latest_vp_pose_left = vp_pose_left
            vp_pose_list_right.append(vp_pose_right)
            vp_pose_list_left.append(vp_pose_left)
        pose_msg_right = numpy_to_pose_stamped(vp_pose_right, frame_id="vp_ground")
        pose_msg_left  = numpy_to_pose_stamped(vp_pose_left, frame_id="vp_ground")
        vp_pub_right.publish(pose_msg_right)
        vp_pub_left.publish(pose_msg_left)
        rate.sleep()

def main():
    global vp_pub_right, vp_pub_left, vp_pose_list_right, vp_pose_list_left
    vp_pose_list_right.clear()
    vp_pose_list_left.clear()

    rospy.init_node('teleop_system_main', anonymous=True)
    
    vp_pub_right = rospy.Publisher("/vision_pro/right_wrist", geometry_msgs.msg.PoseStamped, queue_size=10)
    vp_pub_left  = rospy.Publisher("/vision_pro/left_wrist", geometry_msgs.msg.PoseStamped, queue_size=10)
    
    vision_thread = threading.Thread(target=vision_pro_data_stream)
    vision_thread.daemon = True
    vision_thread.start()
    
    robot, scene, group_right, group_left = init_moveit()
    rospy.loginfo("右臂参考坐标系: " + group_right.get_planning_frame())
    rospy.loginfo("右臂末端: " + group_right.get_end_effector_link())
    rospy.loginfo("左臂参考坐标系: " + group_left.get_planning_frame())
    rospy.loginfo("左臂末端: " + group_left.get_end_effector_link())
    
    rate = rospy.Rate(5) 
    try:
        while not rospy.is_shutdown():
            with data_lock:
                if len(vp_pose_list_right) >= 3:
                    filtered_right = filter_multiple_frames(list(vp_pose_list_right), cutoff=2, fs=10, order=2)
                    current_vp_pose_right = filtered_right[-1]
                else:
                    current_vp_pose_right = latest_vp_pose_right

                if len(vp_pose_list_left) >= 3:
                    filtered_left = filter_multiple_frames(list(vp_pose_list_left), cutoff=2, fs=10, order=2)
                    current_vp_pose_left = filtered_left[-1]
                else:
                    current_vp_pose_left = latest_vp_pose_left

            if current_vp_pose_right is not None:
                T_right = np.eye(4)
                transformed_pose_right = transform_pose(current_vp_pose_right, T_right)
                target_pose_right_np = map_operational_space(transformed_pose_right)
                target_pose_right_msg = numpy_to_pose_stamped(target_pose_right_np, frame_id="base")
                rospy.loginfo("右手规划目标位姿:\n{}".format(target_pose_right_msg))
                plan_and_execute_pose(group_right, target_pose_right_msg)
            else:
                rospy.loginfo("等待右手Vision Pro数据...")

            if current_vp_pose_left is not None:
                T_left = np.eye(4)
                transformed_pose_left = transform_pose(current_vp_pose_left, T_left)
                target_pose_left_np = map_operational_space(transformed_pose_left)
                target_pose_left_msg = numpy_to_pose_stamped(target_pose_left_np, frame_id="base")
                rospy.loginfo("左手规划目标位姿:\n{}".format(target_pose_left_msg))
                plan_and_execute_pose(group_left, target_pose_left_msg)
            else:
                rospy.loginfo("等待左手Vision Pro数据...")

            rate.sleep()
    finally:
        rospy.loginfo("正在关闭 MoveIt 资源...")
        moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
