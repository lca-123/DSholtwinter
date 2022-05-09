import random
import numpy as np
import matplotlib.pyplot as plt
# 双季节指数平滑模型
class DoubleSeasonalHoltWinter():
    """
    双季节指数平滑模型
    """
    def __init__(self,train,test,random_state=None):

        # 平滑项 趋势项 两个季节项 拟合项 预测项
        self.St=[]
        self.Tt=[]
        self.Dt=[]
        self.Wt=[]
        self.y_hats=[]
        self.y_preds=[]

        # 数据集
        self.train=train
        self.test=test

        # 参数
        self.s1=9
        self.s2=7*9
        self.alpha = 0.02
        self.gamma = 0.02
        self.delta = 0.01
        self.omega = 0.2
        self.phi = 0.8

        # 是否进行过预测
        self.is_predict=False

        # 随机数种子
        self.random_state=random_state

    def update_paramter(self,s1=None,s2=None,alpha=None,gamma=None,delta=None,omega=None,phi=None):
        """
        更新参数函数
        """
        if s1 is not None:
            self.s1=s1
        if s2 is not None:
            self.s2=s2
        if alpha is not None:
            self.alpha=alpha   
        if gamma is not None:
            self.gamma=gamma
        if delta is not None:
            self.delta=delta
        if omega is not None:
            self.omega=omega
        if phi is not None:
            self.phi=phi
    
    def _update_paramter_arr(self,param):
        self.update_paramter(alpha=param[0],gamma=param[1],delta=param[2],omega=param[3],phi=param[4])
    
    def _init_state_(self):
        """
        重置状态函数
        """
        # 平滑项 趋势项 两个季节项 拟合项 预测项
        self.St=[]
        self.Tt=[]
        self.Dt=[]
        self.Wt=[]
        self.y_hats=[]
        self.y_preds=[]

        # 设置模型初值
        self.St.append(self.train[0])

        t0=(self.train[-1]-self.train[0])/(len(self.train)-1)
        self.Tt.append(t0)

        for i in range(self.s1):
            si=self.train[i::self.s1]
            self.Dt.append(np.mean(si))

        for i in range(self.s2):
            si=self.train[i::self.s2]
            self.Wt.append(np.mean(si))  

        self.is_predict=False     

    def show_paramter(self):
        """
        显示模型参数
        """
        paramters='alpha='+str(self.alpha)+' ,gamma='+str(self.gamma)+' ,delta='+str(self.delta)+' ,omega='+str(self.omega)+' ,phi='+str(self.phi)
        print(paramters)

    def _fit_(self):
        """
        内置拟合函数，直接使用默认参数拟合模型
        """
        self._init_state_()

        # 拟合模型
        for t in range(len(self.train)):
            st=self.alpha*(self.train[t]-self.Dt[-self.s1]-self.Wt[-self.s2])+(1-self.alpha)*(self.St[-1]+self.Tt[-1])
            self.St.append(st)

            tt=self.gamma*(self.St[-1]-self.St[-2])+(1-self.gamma)*self.Tt[-1]
            self.Tt.append(tt)

            dt=self.delta*(self.train[t]-self.St[-1]-self.Wt[-self.s2])+(1-self.delta)*self.Dt[-self.s1]
            self.Dt.append(dt)

            wt=self.omega*(self.train[t]-self.St[-1]-self.Dt[-self.s1-1])+(1-self.omega)*self.Wt[-self.s2]
            self.Wt.append(wt)

            y_hat=self.St[-1]+self.Tt[-1]+self.Dt[-self.s1]+self.Wt[-self.s2]+self.phi*(self.train[t]-(self.St[-2]+self.Tt[-2]+self.Dt[-self.s1-1]+self.Wt[-self.s2-1]))
            self.y_hats.append(max(0,round(y_hat)))
        
        # self._float2int_('hat')

        # 计算拟合MSE
        mse_fit=np.mean(np.power(np.array(self.train)-np.array(self.y_hats),2))
        return mse_fit
    

    def fit(self,candidates=10,max_iter=10):
        """
        选取模型最优参数拟合模型（使用遗传算法）
        """
        # repeat update_paramter() and fit_MSE()

        if self.random_state is not None:
            random.seed(self.random_state)
        
        # 生成初始参数
        params=[[random.random() for j in range(5)] for i in range(candidates)]
        # print(params)

        # 计算初始适应函数
        fitness_value=[]
        for i in range(candidates):
            param=params[i]
            self._update_paramter_arr(param)
            mse=self._fit_()
            fitness_value.append(mse)
        # print(fitness_value)

        # 迭代
        for iter in range(max_iter):
            # 适应度排序 从高到低两两交叉 选取随机参数乘以随机[0,1]
            index=list(np.argsort(fitness_value))[:int(candidates/2)*2]
            # print(index)
            # print(fitness_value)
            for i in [i for i in range(int(candidates/2)*2)][::2]:
                new_param=list((np.array(params[index[i]])+np.array(params[index[i+1]]))/2)

                # 选取随机参数变异
                _i=random.randint(0,4)
                new_param[_i]=new_param[_i]*random.uniform(0,1/(new_param[_i]+0.2))

                params.append(new_param)
                self._update_paramter_arr(new_param)
                mse=self._fit_()
                fitness_value.append(mse)
        
        # 选取表现最好的作为结果
        index = np.argmin(fitness_value)
        self._update_paramter_arr(params[index])
        mse=self._fit_()

        return mse


        
    def predict(self):
        """
        预测函数
        """
        if self.is_predict==True:
            return('已经预测过!请重新拟合模型再预测')
        for t in range(len(self.test)):
            st=self.alpha*(self.y_hats[-1]-self.Dt[-self.s1]-self.Wt[-self.s2])+(1-self.alpha)*(self.St[-1]+self.Tt[-1])
            self.St.append(st)

            tt=self.gamma*(self.St[-1]-self.St[-2])+(1-self.gamma)*self.Tt[-1]
            self.Tt.append(tt)

            dt=self.delta*(self.y_hats[-1]-self.St[-1]-self.Wt[-self.s2])+(1-self.delta)*self.Dt[-self.s1]
            self.Dt.append(dt)

            wt=self.omega*(self.y_hats[-1]-self.St[-1]-self.Dt[-self.s1-1])+(1-self.omega)*self.Wt[-self.s2]
            self.Wt.append(wt) 

            if t==0:
                y_pred=self.St[-1]+self.Tt[-1]+self.Dt[-self.s1]+self.Wt[-self.s2]+self.phi*(self.train[-1]-(self.St[-2]+self.Tt[-2]+self.Dt[-self.s1-1]+self.Wt[-self.s2-1]))
            else:
                y_pred=self.St[-1]+self.Tt[-1]+self.Dt[-self.s1]+self.Wt[-self.s2]+self.phi*(self.y_preds[-1]-(self.St[-2]+self.Tt[-2]+self.Dt[-self.s1-1]+self.Wt[-self.s2-1]))
            self.y_preds.append(max(0,round(y_pred)))

        self.is_predict=True

        # self._float2int_('pred')

        # 计算预测MSE
        mse_hat=np.mean((np.array(self.test)-np.array(self.y_preds))**2)
        return mse_hat
    
    def _float2int_(self,a):
        """
        将拟合值转变为整数以合理化
        """
        if a=='hat':
            for i in range(len(self.y_hats)):
                self.y_hats[i]=max(0,round(self.y_hats[i]))
        elif a=='pred':
            for i in range(len(self.y_preds)):
                self.y_preds[i]=max(0,round(self.y_preds[i]))            

    def plot_hat(self,t1,t2):
        """
        画出训练集拟合图
        """
        plt.plot(self.y_hats[t1:t2],'r')
        plt.plot(self.train[t1:t2],'b')
        plt.legend(['fit','real'])