import numpy as np
import pandas as pd
from pandas import DataFrame as df
import re
import jieba

def getClassify(classList):
    '''
    说明：自动获取分类表
    
    传参：可迭代对象

    返回值： series 数据
    '''
    return pd.Series(classList,name='ClassifyResult').unique()

def splitDoucumentToWords(doucumnts,stop_words=[]):
    '''
    说明：分词

    传参：
        doucuments：文档（每一行为一篇文档）
        stop_words：不需要的词 (默认为空)
    '''
    regRule = re.compile(r'')

    
    for doucumnt in doucumnts:
        jieba.cut(doucumnt)#默认精确模式，并利用隐马尔可夫
        

pd.DataFrame().content