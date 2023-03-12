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
from scipy import stats
# ------------------------------   slp   ---------------------------
class MapMatrix:
    """
        说明:
            1.构造方法需要两个参数，即二维数组的宽和高
            2.成员变量w和h是二维数组的宽和高
            3.使用:对象[x][y]可以直接取到相应的值
            4.数组的默认值都是0
    """

    def __init__(self, map):
        self.w = map.shape[1]
        self.h = map.shape[0]
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
        self.x = x
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
        if self.map2d[minF.point.y + offsetY][minF.point.x + offsetX] != self.passTag:
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
        if self.map2d[self.endPoint.y][self.endPoint.x] != self.passTag:
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
class SLP:
    """SLP is a hybrid path planning algorithm
    """
    def __init__(self, map2d, obs, s_start, s_goal):
        self.s_start = s_start
        self.s_goal = s_goal
        self.line_slope = np.inf

        self.map2d = map2d  # class Env


        self.reverse = False
        self.g = dict()  # cost to come
        self.astar_sg = [] # astar start and goal ele:[start,goal]

        self.obs = obs
        self.LINEAR_PATH_CALCULATOR()
        # print self.path
        self.BASIS_ALGORITHM_PLANNER()
        self.GENERATE_LINEARIZED_PATH()
        # print self.path

    def LINEAR_PATH_CALCULATOR(self):
        """
        SLP LINEAR_PATH_CALCULATOR.
        :return: path, intersection_points
        """
        path = [(self.s_start[1],self.s_start[0])]
        astar_sg = []
        in_obstacle = False
        # exclude vertical line case
        if self.s_start[0] - self.s_goal[0] != 0:
            self.line_slope = float(self.s_start[1] - self.s_goal[1]) / (self.s_start[0] - self.s_goal[0]) # float division
            # print self.line_slope
            dx_delta = math.copysign(1, self.s_goal[0] - self.s_start[0])
            if dx_delta < 0:
                self.reverse = True
            path_dx = 0
            while True:
                if self.s_start[0] + path_dx == self.s_goal[0]:
                    break
                path_dx += dx_delta
                point = (self.s_start[0] + path_dx, round(self.s_start[1] + self.line_slope * path_dx))
                if point in self.obs:
                    if not in_obstacle:
                        in_obstacle = True
                        a_start = (self.s_start[0] + path_dx - dx_delta, round(self.s_start[1] + self.line_slope * (path_dx - dx_delta)))
                else:
                    if in_obstacle:
                        a_end = point
                        in_obstacle = False
                        astar_sg.append([a_start, a_end])
                    path.append((point[1],point[0]))
            # intersection point processing:



        else:
            path_dy = 0
            dy_delta = math.copysign(1, self.s_goal[1] - self.s_start[1])
            if dy_delta < 0:
                self.reverse = True
            while True:
                if self.s_start[1] + path_dy == self.s_goal[1]:
                    break
                path_dy += dy_delta
                point = (self.s_start[0], self.s_start[1] + path_dy)
                if point in self.obs:
                    if not in_obstacle:
                        a_start = (self.s_start[0], self.s_start[1] + path_dy - dy_delta)

                else:
                    if in_obstacle:
                        a_end = point
                        in_obstacle = False
                        astar_sg.append([a_start, a_end])
                    path.append((point[1],point[0]))
        self.astar_sg = astar_sg
        self.path = path

    def BASIS_ALGORITHM_PLANNER(self):
        """
        USING A*
        """
        slp_path = np.array(self.path)
        for sg in self.astar_sg:
            # print sg
            astar = AStar(self.map2d,Point(int(sg[0][0]),int(sg[0][1])), Point(int(sg[1][0]),int(sg[1][1])))
            a_path = astar.start()
            try:
                a_path = np.array(a_path).reshape(len(a_path),2)
            except Exception:
                continue


            # a_path = a_path[::-1,:] # reverse and remove start and goal
            # print(slp_path)
            # print(a_path)
            if np.where((slp_path == a_path[0]).all(axis=1))[0].shape[0]>0:
                insert_place = np.where((slp_path == a_path[0]).all(axis=1))[0][0]
                slp_path_a = slp_path[0:insert_place]
                slp_path_b = slp_path[min(insert_place+1,slp_path.shape[0]):]
                slp_path = np.concatenate((slp_path_a,a_path))
                slp_path = np.concatenate((slp_path,slp_path_b)) # insert astar path
            elif np.where((slp_path == a_path[-1]).all(axis=1))[0].shape[0]>0:
                insert_place = np.where((slp_path == a_path[-1]).all(axis=1))[0][0]
                slp_path_a = slp_path[0:max(insert_place,0)]
                slp_path_b = slp_path[insert_place:]
                slp_path = np.concatenate((slp_path_a,a_path))
                slp_path = np.concatenate((slp_path,slp_path_b)) # insert astar path
        self.path = slp_path.tolist()

    def PATH_LINEARIZER(self):

        if len(self.path) < 4:
            return self.path

        linearized_path = []

        i = len(self.path) - 1  # parse adversely
        j = i - 1
        pre = j
        linearized_path.append(self.path[i])

        while True:
            # if j == 0:
            #     if self.is_collision(tuple(path[i]), tuple(path[j])):
            #         linearized_path.append(path[pre])
            #     break
            if self.is_collision((self.path[i][1],self.path[i][0]), (self.path[j][1],self.path[j][0])):
                # print("{} and {} collide".format(self.path[i],self.path[j]))
                linearized_path.append(self.path[pre])
                if j == 0:
                    break
                else:
                    i = pre
                    j = pre-1
                    pre = j
            else:
                if j == 0:
                    break
                pre = j
                j -= 1


        linearized_path.append(self.path[0])
        linearized_path = np.array(linearized_path)
        linearized_path = linearized_path[::-1]
        # print(self.path)
        # print(linearized_path[::-1].tolist())
        self.path = linearized_path


    def GENERATE_LINEARIZED_PATH(self):
        pre_path_len = len(self.path)
        while True:
            self.PATH_LINEARIZER()
            cur_path_len = len(self.path)
            if pre_path_len == cur_path_len:
                break
            else:
                pre_path_len = cur_path_len

    def is_collision(self, s_start, s_end, refine_tresh = 7.0):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        # if s_start in self.obs or s_end in self.obs:
        #     print("collided at {} or {}".format(s_start, s_end))
        #     return True

        if s_start[0] - s_end[0] == 0:
            path_dy = 0
            dy_delta = math.copysign(1, s_end[1] - s_start[1])

            while True:
                if s_start[1] + path_dy == s_end[1]:
                    break
                path_dy += dy_delta
                point = (s_start[0], s_start[1] + path_dy)
                if point in self.obs:
                    # print("A collided at {}".format(point))
                    return True
        elif s_start[1] - s_end[1] == 0:
            path_dx = 0
            dx_delta = math.copysign(1, s_end[0] - s_start[0])

            while True:
                if s_start[0] + path_dx == s_end[0]:
                    break
                path_dx += dx_delta
                point = (s_start[0] + path_dx, s_start[1])
                if point in self.obs:
                    # print("B collided at {}".format(point))
                    return True
        else:
            slope = float(s_start[1] - s_end[1]) / (s_start[0] - s_end[0])
            path_dx = 0
            dx_delta = math.copysign(0.25, s_end[0] - s_start[0])

            while True:
                if s_start[0] + path_dx == s_end[0]:
                    break
                path_dx += dx_delta
                round_point = (round(s_start[0] + path_dx), round(s_start[1] + slope * path_dx))
                if round_point in self.obs:
                    return True
        return False

pixwidth = -10
pixheight = -10


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
def getObsMap(map):
    obs_map = set()
    indexList = np.where(map == 1)
    for oy, ox in zip(indexList[0], indexList[1]):
        obs_map.add((ox,oy))
    return obs_map


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

        # 初始化ROS节点
        # rospy.init_node("SLP_global_path_planning", anonymous=True)
        self.reward = 0
        # 将数据处理成一个矩阵（未知：-1，可通行：0，不可通行：1）
        self.doMap()
        # obsize是膨胀系数，是按照矩阵的距离，而不是真实距离，所以要进行一个换算
        self.obsize = 1  # 15太大了
        print("现在进行地图膨胀")
        ob_time = time.time()
        _obstacleMap(self.map, self.obsize)
        print("膨胀地图所用时间是:{:.3f}".format(time.time() - ob_time))
        # self.map_resize()
        obs_map = getObsMap(self.map)
        # 获取初始位置self.init_x,self.init_y
        self.getIniPose()
        # 获取终点位置self.tar_x,self.tar_y
        self.getTarPose()
        print("已接收")

        print(self.width,self.height)
        print("起始点")
        print(self.init_x,self.init_y)
        # print(self.start_point)
        print("目标点")
        print(self.tar_x,self.tar_y)
        print(self.start_point)
        print(self.final_point)

        # #查看是否正确找到起点终点
        # map_test = self.map.copy()
        # map_test[self.start_point[1]][self.start_point[0]] = 1
        # map_test[self.final_point[1]][self.final_point[0]] = 1
        # plt.matshow(map_test, cmap=plt.cm.gray)
        # plt.show()

        # 算法生成
        s_time = time.time()
        self.map2d = MapMatrix(self.map)
        # 创建SLP对象,并设置起点终点
        slp = SLP(self.map2d, obs_map,(self.start_point[1],self.start_point[0]),
                  (self.final_point[1],self.final_point[0]))
        # 开始寻路
        self.pathList = slp.path

        # 查误差
        # print(self.pathList)
        # print("计算之后的终点")
        # print(pixwidth - self.pathList[-1][0]*self.resolution,self.pathList[-1][1]*self.resolution - pixheight)
        # print(self.worldToMap(pixwidth - self.pathList[-1][0]*self.resolution,self.pathList[-1][1]*self.resolution - pixheight))

        print("SLP算法所用时间是:{:.3f}".format(time.time() - s_time))
        # path length
        self.calPathLen()

        # 发布Astar算法
        # self.plotSLPPath()
        # self.runSLPPath()


    # def obstacleMap(self,obsize):
    #     '''
    #     给地图一个膨胀参数

    #     '''

    #     indexList = np.where(self.map == 1)#将地图矩阵中1的位置找到
    #     #遍历地图矩阵

    #     for x in range(self.map.shape[0]):
    #         for y in range(self.map.shape[1]):
    #             if self.map[x][y] == 0:
    #                 for ox,oy in zip(indexList[0],indexList[1]):
    #                     #如果和有1的位置的距离小于等于膨胀系数，那就设为1
    #                     distance = math.sqrt((x-ox)**2+(y-oy)**2)
    #                     if distance <= obsize:
    #                         self.map[x][y] = 1

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
        self.map[self.map == -1] = 0
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

    def plotSLPPath(self):
        plot_map = self.map.copy()
        for i in range(len(self.pathList)):
            y = int(self.pathList[i][0])
            x = int(self.pathList[i][1])
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

    def pubSLPPath(self):
         world_path = []
         for i in range(len(self.pathList)):
            world_path_x = self.pathList[i][0] * self.resolution + pixwidth
            world_path_y = self.pathList[i][1] * self.resolution + pixheight
            world_path.append([world_path_x,world_path_y])
         return world_path

    def worldToMap(self, x, y):
        # 将rviz地图坐标转换为栅格坐标
        # 这里10.2和-4.6需要自动添加，目前不知道怎么添加
        mx = (int)(-(pixwidth - x) / self.resolution)
        my = (int)(-(pixheight - y) / self.resolution)
        return [mx, my]



# ------------------------------       ---------------------------
class TestEnv():
    def __init__(self, action_size):
        self.subgoal_index = 0
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
        try:
            self.global_map = pathPlanning(self.position.x,self.position.y,self.goal_x,self.goal_y).pubSLPPath()[::-1]
        except Exception as e:
            print(e)
            self.global_map = [[self.goal_x,self.goal_y]]
        if math.hypot(self.global_map[0][0]-self.position.x,self.global_map[0][1]-self.position.y) > 0.2:
            print('reversing map')
            self.global_map.reverse()
        if math.hypot(self.global_map[-1][0]-self.goal_x,self.global_map[-1][1]-self.goal_y) > 0.2:
            print('append goal to global map')
            self.global_map.append([self.goal_x,self.goal_y])
        self.global_map = np.array(self.global_map)
        self.subgoal = self.global_map[0]
        self.subgoal_index = 0
        print(self.global_map)
        rospy.loginfo("Sub-Goal position : %.1f, %.1f", self.subgoal[0],
                      self.subgoal[1])


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

        if subgoal_distance < 0.5:
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
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True) #  no respawn when in repeat test mode
            self.goal_distance = self.getGoalDistace()
            print('pausing physics!')
            self.pause_proxy()
            self.generateSubGoals()
            print('unpausing physics!')
            self.unpause_proxy()
            print('successfully unpaused')
            self.subgoal_index = 0
            self.get_goalbox = False

        if self.get_subgoal and self.subgoal_index != self.global_map.shape[0]-1:
            rospy.loginfo("Reached sub-Goal Point")
            self.pub_cmd_vel.publish(Twist())
            self.subgoal_index += 1
            if self.subgoal_index >= self.global_map.shape[0]:
                self.subgoal_index = self.global_map.shape[0]-1
            else:
                rospy.loginfo("Sub-Goal position : %.1f, %.1f", self.subgoal[0],
                          self.subgoal[1])
            self.subgoal = self.global_map[self.subgoal_index]
            self.get_subgoal = False

        return reward

    def getAngleVelDDPG(self, action):  # directional aid
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1) / 2 - action) * max_angular_vel * 0.5
        # if pi/4 > self.heading > -pi/4:  # front
        #     pass
        # elif 3*pi/4 > self.heading >= pi/4:  # left
        #     ang_vel += 0.75
        #     self.action_type = 1
        # elif -pi/4 >= self.heading > -3*pi/4:  # right
        #     ang_vel += -0.75
        #     self.action_type = 2
        # else:  # back
        #     if self.heading >= 3*pi/4:
        #         self.action_type = 3
        #         ang_vel += 1.5
        #     else:
        #         self.action_type = 4
        #         ang_vel += -1.5
        return ang_vel

    def getAngleVelGlobal(self):  # directional aid
        heading = self.heading

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
        # min_range = self.getObstacleMinRange(self.predata)
        # if min_range < 0.46:
        ang_vel = self.getAngleVelDDPG(action)
        vel_cmd.linear.x = 0.15
        # else:
        #     ang_vel = self.getAngleVelGlobal()
        #     if abs(ang_vel)==0.5:
        #         vel_cmd.linear.x = 0.01
        #     else:
        #         vel_cmd.linear.x = 0.15
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

    def get_route(self):
        return [self.pre_x,self.pre_y],[self.cur_x,self.cur_y]

    def reset(self):
        print('resetting environment')
        self.unpause_proxy()
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
        else:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)  # respawn to avoid repeat fails
        print('pausing physics!')
        self.pause_proxy()
        self.generateSubGoals()
        print('unpausing physics!')
        self.unpause_proxy()
        print('successfully unpaused')
        self.subgoal_index = 0

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)