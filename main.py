#抽卡概率分布图.py By OnePoint1.
#本程序用于计算抽卡概率分布图
#但是即便算的再准确，命运
from importlib import reload
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import matrix_power
from collections import defaultdict
#coding = utf-8

#定义初始状态
#==============卡池初始状态=================
InitialCharacterPity = 23
#角色垫了多少抽
InitialWeaponPity = 0
#武器垫了多少抽
IsCharacterGuarantee = 0
#角色池是否为大保底
IsWeaponGuarantee = 0
#武器池是否为大保底
CharacterLostCount = 0
#角色池连歪数
WeaponLostCount = 0
#武器池命定值

#=================抽卡计划=================
WishOrder = 0
#抽取顺序 0：先角色后武器 1：先武器后角色
WishCount = 0
#抽取次数
CharacterCount = 5
#抽角色数量
WeaponCount = 0
#抽武器数量
#========================================

#定义常数
MaxUpCount = CharacterCount + WeaponCount + 2        
#最大Up数
MaxIsWeapon = 2
#是否为武器状态总数
MaxLostCount = 4
#连歪数状态总数
MaxIsGuarantee = 2
#是否大保底状态总数
MaxWishCount = 90
#抽取次数状态总数
MaxStateCount = MaxUpCount * MaxIsWeapon * MaxLostCount * MaxIsGuarantee * MaxWishCount
#状态总数
print("状态总数:", MaxStateCount)
StateTraversed = []
#已遍历状态
P = lil_matrix((MaxStateCount+1, MaxStateCount+1), dtype=np.float64)
#定义转移矩阵
class State:
    #Up数，歪数，是否为武器，是否大保底，概率,
    def __init__(self, UpCount:int, IsWeapon:bool, LostCount:int, IsGuarantee:bool, WishCount:int, Probability:float):
        #UpCount: 0+, IsWeapon: 0-1, LostCount: 0-3, IsGuarantee: 0-1, WishCount: 0-90/0-80
        self.UpCount = UpCount
        self.IsWeapon = IsWeapon
        self.LostCount = LostCount
        self.IsGuarantee = IsGuarantee
        self.WishCount = WishCount
        self.probability = Probability
        '''
        if self.UpCount >= MaxUpCount or self.IsWeapon >= MaxIsWeapon-1 or self.LostCount > MaxLostCount-1 or self.IsGuarantee > MaxIsGuarantee-1 or self.WishCount > MaxWishCount-1:
            raise ValueError("状态参数超出范围")
        if self.UpCount < 0 or self.IsWeapon < 0 or self.LostCount < 0 or self.IsGuarantee < 0 or self.WishCount < 0:
            raise ValueError("状态参数不能为负数")
        '''        
        if self.UpCount == CharacterCount and self.IsWeapon == 0:
            self.LostCount = 0
            self.IsGuarantee = 0
        if self.UpCount == WeaponCount and self.IsWeapon == 1:
            self.LostCount = 0
            self.IsGuarantee = 0
        #到达预期 矫正状态
    def ToMatrixIndex(self)->int:
        return (((self.UpCount * MaxIsWeapon + self.IsWeapon) * MaxLostCount + self.LostCount )* MaxIsGuarantee + self.IsGuarantee )* MaxWishCount + self.WishCount
    def ToIndex(self)->int:
        return ((self.UpCount * MaxIsWeapon + self.IsWeapon ) * MaxLostCount + self.LostCount ) * MaxIsGuarantee + self.IsGuarantee
    def Check(self)->bool:
        return True
    def __str__(self)->str: 
        return self.UpCount*1000 + self.IsWeapon*100 + self.LostCount*10 + self.IsGuarantee
    


def Next(CurrentState:State)-> None:
    if CurrentState.__str__() in StateTraversed:
        return
        #已遍历过该状态，直接返回
    else:
        StateTraversed.append(CurrentState.__str__())
        #新增已遍历状态
        if CurrentState.IsWeapon == 0:
            #抽角色
            if CurrentState.UpCount < CharacterCount - 1:  
                #下一抽未达预期，
                if CurrentState.IsGuarantee == 1:
                    #大保底抽取
                    Next(State(CurrentState.UpCount + 1, 0, CurrentState.LostCount, 0, 0, 1))
                elif CurrentState.LostCount == 3:
                    #已连歪三次，100%概率出角色
                    Next(State(CurrentState.UpCount + 1, 0, 1, 0, 0, 1))
                else:
                    if CurrentState.LostCount == 2:
                        #已连歪两次，75%概率出角色
                        pass
                    #正常抽取
                    Next(State(CurrentState.UpCount + 1, 0, max(0,CurrentState.LostCount-1), 0, 0, 0.5))
                    #抽中up角色
                    Next(State(CurrentState.UpCount, 0, CurrentState.LostCount + 1, 1, 0, 0.5))
                    #歪了
                    #fill CurrentState State1 State2 p1 p2
                    #print("当前状态:", State(CurrentState.UpCount, 0, CurrentState.LostCount + 1, 1, 0, 0.5).__str__())
            elif CurrentState.UpCount == CharacterCount-1:
                #最后一个金大保底
                if CurrentState.IsGuarantee != 1:
                    #下一个金不是大保底
                    Next(State(CurrentState.UpCount, 0, CurrentState.LostCount+1, 1, 0, 0))
                if WeaponCount > 0 and WishOrder == 0:
                    #已达预期抽武器 抽取顺序
                    if CurrentState.IsGuarantee != 1:
                        pass
                    Next(State(0, 1, WeaponLostCount, IsWeaponGuarantee, InitialWeaponPity, 1))
                else:
                    #无武器抽取计划结束
                    if CurrentState.IsGuarantee != 1:
                        pass
                    Next(State(MaxUpCount-2,0,0,0,0,0))
                    print("抽卡结束")
            pass
        else:
            #抽武器
            if CurrentState.UpCount < WeaponCount - 1:  
                #下一抽未达预期，
                if CurrentState.LostCount == 1:
                    #命定值满
                    Next(State(CurrentState.UpCount + 1, 1, 0, 0, 0, 1))
                else:
                    if CurrentState.IsGuarantee == 1 and CurrentState.UpCount == 0:
                        #初始前一抽不是up,那么二选一
                        Next(State(CurrentState.UpCount + 1, 0, 0, 0, 0, 0.5))
                        Next(State(CurrentState.UpCount ,1, 1, 0, 0, 0.5))
                        pass
                    #正常抽取
                    else:
                        Next(State(CurrentState.UpCount + 1, 1, 0, 0, 0, 0.375))
                        #抽中up角色
                        Next(State(CurrentState.UpCount, 1, 1, 1, 0, 0.625))
                        #歪了
                    #print("当前状态:", State(CurrentState.UpCount, 0, CurrentState.LostCount + 1, 1, 0, 0.5).__str__())
            elif CurrentState.UpCount == WeaponCount-1:
                #最后一个金大保底
                if CurrentState.LostCount != 1:
                    #下一个金不是大保底
                    Next(State(CurrentState.UpCount, 1, 1, 1, 0, 0))
                if CharacterCount > 0 and WishOrder == 1:
                    #已达预期抽武器 抽取顺序
                    if CurrentState.LostCount != 1:
                        pass
                    #Next(State(0, 1, WeaponLostCount, IsWeaponGuarantee, InitialWeaponPity, 1))
                else:
                    #无武器抽取计划结束
                    if CurrentState.LostCount != 1:
                        pass
                    Next(State(MaxUpCount-2,1,0,0,0,0))
                    print("抽卡结束")
                pass
            pass
            pass


print(State(MaxUpCount,MaxIsWeapon-1,MaxLostCount-1,MaxIsGuarantee-1,MaxWishCount,0).__str__())
print(State(0,1,0,1,0,0).__str__())
CharacterStart = State(0, 0, CharacterLostCount, IsCharacterGuarantee, InitialCharacterPity, 0)
Next(CharacterStart)
#开始遍历抽卡状态
print(StateTraversed)
