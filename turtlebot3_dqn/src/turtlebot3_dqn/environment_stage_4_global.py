#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn

class Env():
    def __init__(self):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.initGoal = True
        self.get_goalbox = False
        self.get_point = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.yaw = yaw

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi
        # ensure heading is in [-pi,pi]
        self.heading = round(heading, 2)

    def getState(self, scan, point):
        scan_range = []
        min_range = 0.13
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        if current_distance < 0.2:
            self.get_goalbox = True

        point_distance = round(math.hypot(point[0] - self.position.x, point[1] - self.position.y), 2)

        if point_distance < 0.05:
            self.get_point = True

        return done

    def setReward(self,done):
        reward = 0
        if done:
            rospy.loginfo("Collision!!")
            reward = -500
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            reward = 1000
            rospy.loginfo("Goal!!")
            self.pub_cmd_vel.publish(Twist())
            # self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)

        if self.get_point:
            rospy.loginfo("Reached Path Point")
            self.pub_cmd_vel.publish(Twist())
        return reward

    def getAngleVel(self, point):  # directional aid
        point_angle = math.atan2(point[1] - self.position.y, point[0] - self.position.x)

        heading = point_angle - self.yaw

        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi
        # ensure heading is in [-pi,pi]

        if abs(heading) > 0.6:
            ang_vel = math.copysign(0.5,heading)
        else:
            ang_vel = math.copysign(0.1,heading)

        return ang_vel


    def step(self, point):
        # max_angular_vel = 1.5
        # ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5
        self.get_point = False
        reward = 0
        route = []
        while not self.get_point:
            route.append([self.position.x,self.position.y])
            ang_vel = self.getAngleVel(point)

            vel_cmd = Twist()
            if abs(ang_vel)==0.5:
                vel_cmd.linear.x = 0.01
            else:
                vel_cmd.linear.x = 0.15
            vel_cmd.angular.z = ang_vel
            self.pub_cmd_vel.publish(vel_cmd)

            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message('scan', LaserScan, timeout=5)
                except:
                    pass
            done = self.getState(data,point)
            reward += self.setReward(done)
            if done or self.get_point or self.get_goalbox:
                break

        return reward, done ,route


    def reset(self):
        print('resetting environment!')
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass


        self.goal_x, self.goal_y = self.respawn_goal.getPosition(True,True)


        done = self.getState(data, (self.goal_x,self.goal_y))

        return done