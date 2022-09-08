#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

import os
import json
import random
import time
import sys
from numba import jit
import matplotlib.pyplot as plt
from nav_msgs.msg import OccupancyGrid

# ------------------------------   astar   ---------------------------

class MapMatrix:
    """
        说明:
            1.构造方法需要两个参数，即二维数组的宽和高
            2.成员变量w和h是二维数组的宽和高
            3.使用:对象[x][y]可以直接取到相应的值
            4.数组的默认值都是0
    """

    def __init__(self, map):
        self.w = map.shape[0]
        self.h = map.shape[1]
        self.data = map

    def showArrayD(self):
        for y in range(self.h):
            for x in range(self.w):
                print(self.data[x][y],)
            print("")

    def __getitem__(self, item):
        return self.data[item]


class Point:
    """
    表示一个点
    """

    def __init__(self, x, y):
        self.x = x;
        self.y = y

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        return False
    # def __str__(self):
    #     #return "x:"+str(self.x)+",y:"+str(self.y)
    #     return [self.y,self.x]


class AStar:
    """
    AStar算法的Python3.x实现
    """

    class Node:  # 描述AStar算法中的节点数据
        def __init__(self, point, endPoint, g=0):
            self.point = point  # 自己的坐标
            self.father = None  # 父节点
            self.g = g  # g值，g值在用到的时候会重新算
            self.h = (abs(endPoint.x - point.x) + abs(endPoint.y - point.y)) * 10  # 计算h值

    def __init__(self, map2d, startPoint, endPoint, passTag=0):
        """
        构造AStar算法的启动条件
        :param map2d: ArrayD类型的寻路数组
        :param startPoint: Point或二元组类型的寻路起点
        :param endPoint: Point或二元组类型的寻路终点
        :param passTag: int类型的可行走标记（若地图数据!=passTag即为障碍）
        """
        # 开启表
        self.openList = []
        # 关闭表
        self.closeList = []
        # 寻路地图
        self.map2d = map2d
        # 起点终点
        if isinstance(startPoint, Point) and isinstance(endPoint, Point):
            self.startPoint = startPoint
            self.endPoint = endPoint
        else:
            self.startPoint = Point(*startPoint)
            self.endPoint = Point(*endPoint)

        # 可行走标记
        self.passTag = passTag

    def getMinNode(self):
        """
        获得openlist中F值最小的节点
        :return: Node
        """
        currentNode = self.openList[0]
        for node in self.openList:
            if node.g + node.h < currentNode.g + currentNode.h:
                currentNode = node
        return currentNode

    def pointInCloseList(self, point):
        for node in self.closeList:
            if node.point == point:
                return True
        return False

    def pointInOpenList(self, point):
        for node in self.openList:
            if node.point == point:
                return node
        return None

    def endPointInCloseList(self):
        for node in self.openList:
            if node.point == self.endPoint:
                return node
        return None

    def searchNear(self, minF, offsetX, offsetY):
        """
        搜索节点周围的点
        :param minF:F值最小的节点
        :param offsetX:坐标偏移量
        :param offsetY:
        :return:
        """
        # 越界检测
        if minF.point.x + offsetX < 0 or minF.point.x + offsetX > self.map2d.w - 1 or minF.point.y + offsetY < 0 or minF.point.y + offsetY > self.map2d.h - 1:
            return
        # 如果是障碍，就忽略
        if self.map2d[minF.point.x + offsetX][minF.point.y + offsetY] != self.passTag:
            return
        # 如果在关闭表中，就忽略
        currentPoint = Point(minF.point.x + offsetX, minF.point.y + offsetY)
        if self.pointInCloseList(currentPoint):
            return
        # 设置单位花费
        if offsetX == 0 or offsetY == 0:
            step = 10
        else:
            step = 14
        # 如果不再openList中，就把它加入openlist
        currentNode = self.pointInOpenList(currentPoint)
        if not currentNode:
            currentNode = AStar.Node(currentPoint, self.endPoint, g=minF.g + step)
            currentNode.father = minF
            self.openList.append(currentNode)
            return
        # 如果在openList中，判断minF到当前点的G是否更小
        if minF.g + step < currentNode.g:  # 如果更小，就重新计算g值，并且改变father
            currentNode.g = minF.g + step
            currentNode.father = minF

    def start(self):
        """
        开始寻路
        :return: None或Point列表（路径）
        """
        # 判断寻路终点是否是障碍
        if self.map2d[self.endPoint.x][self.endPoint.y] != self.passTag:
            return None

        # 1.将起点放入开启列表
        startNode = AStar.Node(self.startPoint, self.endPoint)
        self.openList.append(startNode)
        # 2.主循环逻辑
        while True:
            # 找到F值最小的点
            minF = self.getMinNode()
            # 把这个点加入closeList中，并且在openList中删除它
            self.closeList.append(minF)
            self.openList.remove(minF)
            # 判断这个节点的上下左右节点
            self.searchNear(minF, 0, -1)
            self.searchNear(minF, 0, 1)
            self.searchNear(minF, -1, 0)
            self.searchNear(minF, 1, 0)
            # 判断是否终止
            point = self.endPointInCloseList()
            if point:  # 如果终点在关闭表中，就返回结果
                # print("关闭表中")
                cPoint = point
                pathList = []
                while True:
                    if cPoint.father:
                        # pathList.append(cPoint.point)
                        pathList.append([cPoint.point.y, cPoint.point.x])
                        cPoint = cPoint.father
                    else:
                        # print(pathList)
                        # print(list(reversed(pathList)))
                        # print(pathList.reverse())
                        return list(reversed(pathList))
            if len(self.openList) == 0:
                return None


# ------------------------------   ros路径规划配置   ---------------------------
# 这个需要根据自己的地图而定
pixwidth = 2.425  # 10.2
pixheight = 2.425  # 4.6


# 将最慢算法的加速一下
@jit(nopython=True)
def _obstacleMap(map, obsize):
    '''
    给地图一个膨胀参数

    '''

    indexList = np.where(map == 1)  # 将地图矩阵中1的位置找到
    # 遍历地图矩阵

    for x in range(map.shape[0]):
        for y in range(map.shape[1]):
            if map[x][y] == 0:
                for ox, oy in zip(indexList[0], indexList[1]):
                    # 如果和有1的位置的距离小于等于膨胀系数，那就设为1
                    distance = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                    if distance <= obsize:
                        map[x][y] = 1


class pathPlanning():
    def __init__(self,start_x,start_y,goal_x,goal_y):
        '''
        起点:[2,2]
        终点:[2,4]
        地图:（未知:-1，可通行:0，不可通行:1）


        返回的内容:[(2,4),(1,4),(0,3),(1,2),(2,2)]
        '''

        # 初始化ROS节点
        # rospy.init_node("Astar_global_path_planning", anonymous=True)

        self.init_x = start_x
        self.init_y = start_y
        self.tar_x = goal_x
        self.tar_y = goal_y

        self.reward = 0
        # 将数据处理成一个矩阵（未知：-1，可通行：0，不可通行：1）
        self.doMap()
        # obsize是膨胀系数，是按照矩阵的距离，而不是真实距离，所以要进行一个换算
        self.obsize = 4.5  # 15太大了
        print("现在进行地图膨胀")
        ob_time = time.time()
        _obstacleMap(self.map, self.obsize)
        print("膨胀地图所用时间是:{:.3f}".format(time.time() - ob_time))
        self.map_resize()
        # 获取初始位置self.init_x,self.init_y
        self.getIniPose()
        # 获取终点位置self.tar_x,self.tar_y
        self.getTarPose()
        print("已接收")

        # print(self.width,self.height)
        # print("起始点")
        # print(self.init_x,self.init_y)
        # print(self.start_point)
        # print("目标点")
        # print(self.tar_x,self.tar_y)
        # print(self.start_point[0])
        # print(self.final_point)

        # #查看是否正确找到起点终点
        # map_test = self.map.copy()
        # map_test[self.start_point[1]][self.start_point[0]] = 1
        # map_test[self.final_point[1]][self.final_point[0]] = 1
        # plt.matshow(map_test, cmap=plt.cm.gray)
        # plt.show()

        # 算法生成
        s_time = time.time()
        self.map2d = MapMatrix(self.map)
        # 创建AStar对象,并设置起点终点
        aStar = AStar(self.map2d, Point(self.start_point[1], self.start_point[0]),
                      Point(self.final_point[1], self.final_point[0]))
        # 开始寻路
        self.pathList = aStar.start()

        # 查误差
        # print(self.pathList)
        # print("计算之后的终点")
        # print(pixwidth - self.pathList[-1][0]*self.resolution,self.pathList[-1][1]*self.resolution - pixheight)
        # print(self.worldToMap(pixwidth - self.pathList[-1][0]*self.resolution,self.pathList[-1][1]*self.resolution - pixheight))

        print("Astar算法所用时间是:{:.3f}".format(time.time() - s_time))
        # path length
        # self.calPathLen()

        # 发布Astar算法
        # self.pubAstarPath()

    def doMap(self):
        '''
            获取数据
            将数据处理成一个矩阵（未知:-1，可通行:0，不可通行:1）
        '''
        # 获取地图数据
        self.OGmap = rospy.wait_for_message("/map", OccupancyGrid, timeout=None)
        # 地图的宽度
        self.width = self.OGmap.info.width
        # 地图的高度
        self.height = self.OGmap.info.height
        # 地图的分辨率
        self.resolution = self.OGmap.info.resolution

        # 获取地图的数据 可走区域的数值为0，障碍物数值为100，未知领域数值为-1
        mapdata = np.array(self.OGmap.data, dtype=np.int8)
        # 将地图数据变成矩阵
        self.map = mapdata.reshape((self.height, self.width))

        # 将地图中的障碍变成从100变成1
        self.map[self.map == 100] = 1
        # 列是逆序的，所以要将列顺序
        self.map = self.map[:, ::-1]

        # # #查看地图数据存储格式
        # plt.matshow(self.map, cmap=plt.cm.gray)
        # plt.show()
    def map_resize(self):
        # resize
        indexList = np.where(self.map == 1)  # 将地图矩阵中1的位置找到
        upper_left = (min(indexList[0]), min(indexList[0]))
        lower_rignt = (max(indexList[0]), max((indexList[1])))

        self.map = self.map[upper_left[0]:lower_rignt[0], upper_left[1]:lower_rignt[1]]
        # plt.matshow(self.map, cmap=plt.cm.gray)
        # plt.show()

    def getIniPose(self):
        '''
            获取初始坐标点
        '''
        # 获取对于矩阵中的原始点位置
        self.start_point = self.worldToMap(self.init_x, self.init_y)

    def getTarPose(self):
        '''
            获取目标坐标点
        '''
        self.final_point = self.worldToMap(self.tar_x, self.tar_y)

    def plotAstarPath(self):
        plot_map = self.map.copy()
        for i in range(len(self.pathList)):
            y = self.pathList[i][0]
            x = self.pathList[i][1]
            plot_map[x,y] = 2
        plt.matshow(plot_map, cmap=plt.cm.gray)
        plt.show()

    def calPathLen(self):
        length = 0
        for i in range(len(self.pathList)-1):
            world_path_x_1 = pixwidth - self.pathList[i][0] * self.resolution
            world_path_y_1 = self.pathList[i][1] * self.resolution - pixheight
            world_path_x_2 = pixwidth - self.pathList[i+1][0] * self.resolution
            world_path_y_2 = self.pathList[i+1][1] * self.resolution - pixheight
            length += math.hypot(world_path_x_1-world_path_x_2,world_path_y_1-world_path_y_2)
        self.pathLength = length

    def pubAstarPath(self):
        world_path = []
        for i in range(len(self.pathList)):
            world_path_x = pixwidth - self.pathList[i][0] * self.resolution
            world_path_y = self.pathList[i][1] * self.resolution - pixheight
            world_path.append([world_path_x,world_path_y])
        return world_path


    def worldToMap(self, x, y):
        # 将rviz地图坐标转换为栅格坐标
        # 这里10.2和-4.6需要自动添加，目前不知道怎么添加
        mx = (int)((pixwidth - x) / self.resolution)
        my = (int)(-(-pixheight - y) / self.resolution)
        return [mx, my]


# ------------------------------       ---------------------------
class TestEnv():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.get_subgoal = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.action_type = 0  # 0:front 1:left 2:right 3:back_l 4:back_r

        self.subgoal = [0,0]

    def generateSubGoals(self):
        print("now generating sub-goals")
        self.global_map = np.array(pathPlanning(self.position.x,self.position.y,self.goal_x,self.goal_y).pubAstarPath())
        self.subgoal = self.global_map[0]
        self.subgoal_index = 0

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.yaw = yaw
        # goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        goal_angle = math.atan2(self.subgoal[1] - self.position.y, self.subgoal[0] - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi
        # ensure heading is in [-pi,pi]
        self.heading = round(heading, 2)

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.13
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)
        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance < 0.2:
            self.get_goalbox = True

        subgoal_distance = round(math.hypot(self.subgoal[0] - self.position.x, self.subgoal[1] - self.position.y), 2)

        if subgoal_distance < 0.15:
            self.get_subgoal = True

        return scan_range + [heading, subgoal_distance, obstacle_min_range, obstacle_angle], done

    def setReward(self, state, done, action):
        reward = 0

        if done:
            rospy.loginfo("Collision!!")
            reward = -500
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1000
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.generateSubGoals()
            self.get_goalbox = False

        if self.get_subgoal:
            rospy.loginfo("Reached sub-Goal Point")
            self.pub_cmd_vel.publish(Twist())
            self.subgoal_index += 1
            if self.subgoal_index >= self.global_map.shape[0]:
                self.subgoal_index = self.global_map.shape[0]-1
            self.subgoal = self.global_map[self.subgoal_index]
            self.get_subgoal = False

        return reward

    def getAngleVelDDPG(self, action):  # directional aid
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1) / 2 - action) * max_angular_vel * 0.5
        if pi/4 > self.heading > -pi/4:  # front
            pass
        elif 3*pi/4 > self.heading >= pi/4:  # left
            ang_vel += 0.75
            self.action_type = 1
        elif -pi/4 >= self.heading > -3*pi/4:  # right
            ang_vel += -0.75
            self.action_type = 2
        else:  # back
            if self.heading >= 3*pi/4:
                self.action_type = 3
                ang_vel += 1.5
            else:
                self.action_type = 4
                ang_vel += -1.5
        return ang_vel

    def getAngleVelGlobal(self):  # directional aid
        point_angle = math.atan2(self.subgoal[1] - self.position.y, self.subgoal[0] - self.position.x)

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

    def getObstacleMinRange(self,scan):
        scan_range = []

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)

        return obstacle_min_range

    def step(self, action):
        # max_angular_vel = 1.5
        # ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        self.pre_x = self.position.x
        self.pre_y = self.position.y

        vel_cmd = Twist()
        min_range = self.getObstacleMinRange(self.predata)
        if min_range < 0.25:
            ang_vel = self.getAngleVelDDPG(action)
            vel_cmd.linear.x = 0.15
        else:
            ang_vel = self.getAngleVelGlobal()
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

        self.predata = data
        state, done = self.getState(data)
        reward = self.setReward(state, done, action)
        self.cur_x = self.position.x
        self.cur_y = self.position.y
        pathlen = math.hypot((self.cur_y-self.pre_y),(self.cur_x-self.pre_x))

        return np.asarray(state), reward, done, pathlen

    def reset(self):
        print('resetting environment')
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
        self.predata = data
        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False
        print('pausing physics!')
        self.pause_proxy()
        self.generateSubGoals()
        print('unpausing physics!')
        self.unpause_proxy()
        print('successfully unpaused')

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)