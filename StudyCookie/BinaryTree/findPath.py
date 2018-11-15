'''
说明：
    回想了一下python建树的过程，对于二叉树，路径查找，我们可以考虑list 的字符建树
'''
import copy 
class TreeErr(Exception):
    def __init__(self):
        self.value = '建树错误！，请检查您的 树'
    def __str__(self):
        return repr(self.value)
class FindPath():
    
    def __init__(self,originalData):
        '''
        初始化：
            
        传参：
            originalTree: 原始数据 list of lists 的变形形式 
                          例：样式【 根 ，'[' , 左子树 ,']' ,'[' , 右子树 ,']'】
                          说明：整个非层级结构，而是从左到右的元素，用 字符 ‘[’ , ']'去匹配，
                          *启发来源，面试中的字符匹配
        '''
        if self.checkTree(originalData) == False:
            raise TreeErr()

        self.originalData = originalData
    
    
    def find(self,aimNumber):
        '''
        找路径

        思想：字符匹配

        输入：
            aimNumber : 目标数字
        返回：
            第一次的匹配路径 list
        '''
        dataTree = copy.deepcopy(self.originalData) #保护原数据
        roadPath = [] #路径数据，
        reduceNum = 0#每次dataTree出栈后，需要调整i 不然会下标溢出

        for i in range(len(dataTree)):
            index = i - reduceNum
            d = dataTree[index]
            if d == '(':
                roadPath.append(dataTree[index+1])#更新list 路径的入栈
            elif d == ')':

                if sum(roadPath) == aimNumber:#相等输出
                    return roadPath #
                else :
                    roadPath.pop() #路径的出栈

                del dataTree[index-2:index+1]
                reduceNum += 3 #每次删除三个数据

        return "Sorry ! Can't find"

    def checkTree(self,originalData):
        '''
        检查树是否正确 更新rightData状态

        *待加异常

        True 正确/ False错误
        '''
        count = 0
        for d in originalData:
            if d == '[':
                count += 1
            if d == ']':
                count -=1
        return count == 0

if __name__ == '__main__':
    tree = ['(',1,'(',2,'(',3,'(',1,')','(',4,')',')','(',0,')',')',')']
    # print(FindPath.checkTree(tree))
    t = FindPath(tree)
    print(t.find(6))
    # print(t.find(10))