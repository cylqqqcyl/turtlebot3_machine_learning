#! /usr/bin/env python


import time
from numba import jit
import math
import rospy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


# ------------------------    Astar     --------------------------
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
                print(self.data[x][y], end=' ')
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
    def __init__(self):
        '''
        起点:[2,2]
        终点:[2,4]
        地图:（未知:-1，可通行:0，不可通行:1）


        返回的内容:[(2,4),(1,4),(0,3),(1,2),(2,2)]
        '''

        # 初始化ROS节点
        rospy.init_node("Astar_global_path_planning", anonymous=True)

        # 将数据处理成一个矩阵（未知：-1，可通行：0，不可通行：1）
        self.doMap()
        # obsize是膨胀系数，是按照矩阵的距离，而不是真实距离，所以要进行一个换算
        self.obsize = 7  # 15太大了
        print("现在进行地图膨胀")
        ob_time = time.time()
        _obstacleMap(self.map, self.obsize)
        print("膨胀地图所用时间是:{:.3f}".format(time.time() - ob_time))

        # 获取初始位置self.init_x,self.init_y
        self.getIniPose()
        # 获取终点位置self.tar_x,self.tar_y
        self.getTarPose()
        print("已接收")

        # print(self.width,self.height)
        # print("起始点")
        # print(self.init_x,self.ros中init_y)
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
        # print("计算之后的终点")
        # print(pixwidth - self.pathList[-1][0]*self.resolution,self.pathList[-1][1]*self.resolution - pixheight)
        # print(self.worldToMap(pixwidth - self.pathList[-1][0]*self.resolution,self.pathList[-1][1]*self.resolution - pixheight))

        print("Astar算法所用时间是:{:.3f}".format(time.time() - s_time))

        # 发布Astar算法
        self.sendAstarPath()

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
        # 列是逆序的，所以要将列顺序
        self.map = self.map[:, ::-1]

        # #查看地图数据存储格式
        plt.matshow(self.map, cmap=plt.cm.gray)
        plt.show()

    def getIniPose(self):
        '''
            获取初始坐标点
        '''
        self.IniPose = rospy.wait_for_message("/amcl_pose", PoseWithCovarianceStamped, timeout=None)
        self.init_x = self.IniPose.pose.pose.position.x
        self.init_y = self.IniPose.pose.pose.position.y
        # 获取对于矩阵中的原始点位置
        self.start_point = self.worldToMap(self.init_x, self.init_y)

        self.init_quaternions_z = self.IniPose.pose.pose.orientation.z
        self.init_quaternions_w = self.IniPose.pose.pose.orientation.w

    def getTarPose(self):
        '''
            获取目标坐标点
        '''
        self.TarPose = rospy.wait_for_message("/move_base_simple/goal", PoseStamped, timeout=None)
        self.tar_x = self.TarPose.pose.position.x
        self.tar_y = self.TarPose.pose.position.y
        self.final_point = self.worldToMap(self.tar_x, self.tar_y)
        self.tar_quaternions_x = self.TarPose.pose.orientation.x
        self.tar_quaternions_y = self.TarPose.pose.orientation.y
        self.tar_quaternions_z = self.TarPose.pose.orientation.z
        self.tar_quaternions_w = self.TarPose.pose.orientation.w

    def sendAstarPath(self):
        AstarPath = rospy.Publisher("AstarPath", Path, queue_size=15)
        init_path = Path()

        # 设置发布频率
        rate = rospy.Rate(200)

        for i in range(len(self.pathList)):
            init_path.header.stamp = rospy.Time.now()
            init_path.header.frame_id = "map"

            current_point = PoseStamped()
            current_point.header.frame_id = "map"
            current_point.pose.position.x = pixwidth - self.pathList[i][0] * self.resolution
            current_point.pose.position.y = self.pathList[i][1] * self.resolution - pixheight
            current_point.pose.position.z = 0
            # 角度
            current_point.pose.orientation.x = 0
            current_point.pose.orientation.y = 0
            current_point.pose.orientation.z = 0
            current_point.pose.orientation.w = 1

            init_path.poses.append(current_point)
            # 发布消息
            AstarPath.publish(init_path)

            rate.sleep()
            i += 1
        time.sleep(0.5)

    def worldToMap(self, x, y):
        # 将rviz地图坐标转换为栅格坐标
        # 这里10.2和-4.6需要自动添加，目前不知道怎么添加
        mx = (int)((pixwidth - x) / self.resolution)
        my = (int)(-(-pixheight - y) / self.resolution)
        return [mx, my]


if __name__ == "__main__":
    getmap = pathPlanning()
