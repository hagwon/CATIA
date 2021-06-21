# iXjXk의 일반화 신경망을 만들어 보자. 오미입, 오메가 미입 공식 활용
# 출력은 항등함수, 은닉층은 tanh로 활성화 (정규화 불필요). 
# 단, 항등함수는 학습데이터가 많거나 패턴이 복잡하면 수렴을 못하니 주의!!!

import numpy as np
from random import random
import matplotlib.pyplot as plt

######################### Hyper Parameter ######################
alpha = 0.1         # 학습율 (Learning Rate)
epoch = 1000        # 학습 에포크(학습 횟수)
n_hidden = 5        # 은닉층 노드수

############ 가중치 초기화 함수 (weight initialize) ############# 
wt = []           # 가중치의 빈 list (vacant array for weights)
bs = []           # bias의 빈 list (vacant array for bias)
def init_weight(n_input, n_output):
    global wt, bs
    for i in range(n_input*n_hidden + n_hidden*n_output):
        w = np.random.rand()
        wt.append(w)
    for i in range(n_hidden + n_output):
        w = np.random.rand()
        bs.append(w)

######## 하이퍼탄젠트 활성화 함수 (tanh activation function) ###### 
def tanh(x):
    y = np.tanh(x)
    return y

######################### 순방향(Forward) 계산 ######################
def forward(x, n_output):
    u = []; y = []
    n_input = len(x)
    # 은닉층 출력값
    for j in range(n_hidden):        # 은닉층 노드 갯수
        sum = 0
        for n in range(n_input):     # 입력 갯수
            tmp = wt[n*n_hidden+j] * x[n] 
            sum = sum + tmp
        u.append(tanh(sum + bs[j]))

    # 출력층 출력값
    for k in range(n_output):
        sum = 0
        for n in range(n_hidden):
            tmp = wt[n_input*n_hidden + n*n_output+k] * u[n]
            sum = sum + tmp
        y.append(sum+bs[n_hidden+k])

    return u, y

#################### 역방향(기울기) 계산 (Back Propagation) #################
def backpropagate(x, u, y, t):
    dE_dw = []          # 가중치 기울기 값의 빈 list
    dE_db = []          # Bias 기울기 값의 빈 list
    n_input = len(x); n_output = len(t)
    for i in range(n_input):
        for j in range(n_hidden):
            sum = 0
            for n in range(n_output):
                tmp = (y[n]-t[n])*wt[n_input*n_hidden+j+n_hidden*n] # 오메가미입
                # "오메가" 부분을 우선적으로 계산
                sum = sum + tmp
            dE_dw.append(sum*(1-u[j])*(1+u[j])*x[i])
                # 은칙층의 기울기 공식: 오메가 x 미입

    for j in range(n_hidden):
        sum = 0
        for k in range(n_output):
            dE_dw.append((y[k]-t[k])*u[j])  # 오미입
            # 출력층의 기울기 공식: 오미입
            tmp = (y[k]-t[k])*wt[n_input*n_hidden+j+n_hidden*k]
            sum = sum + tmp
        dE_db.append(sum*(1-u[j])*(1+u[j]))
            # 은닉층 bias에 대한 기울기 공식: 오메가 x 미입 (입력은 1)
    
    for i in range(n_output):
        tmp = (y[i]-t[i])
        dE_db.append(tmp)
            # 출력층 bias에 대한 기울기 공식: 오미입 (입력은 1)
    
    return dE_dw, dE_db

################## 최적화 (걍사하강법) 계산 (Gradient Decent) #################
def update_weight(dE_dw, dE_db):
    global wt, bs
    for i in range(len(wt)):
        wt[i] = wt[i] - alpha * dE_dw[i]
    for i in range(len(bs)):
        bs[i] = bs[i] - alpha * dE_db[i]

###################### 손실함수 (Loss Function) 계산 #######################
def calc_error(y, t):
    err = 0
    for i in range(len(t)):
        tmp = 0.5*(y[i]-t[i])**2
        err = err + tmp
    return err

def error_graph(error):  # 학습이 진행됨(epoch)에 따른 에러값 변화를 가시화
    plt.ylim(0.0, 1)
    plt.plot(np.arange(0, error.shape[0]), error)
    plt.show()

####################  신경망으로 학습 (Learning by NN)  ###################
def train(X, T):

    error = np.zeros(epoch)            # 손실함수(오차) 초기화

    n_input = X.shape[1]               # 입력 노드수
    n_output = T.shape[1]              # 출력 노드수

    # 가중치 초기화
    init_weight(n_input, n_output)

    ###### 입력과 정답으로 학습 (train with input and teaching datum) ######
    for n in range(epoch):                  # epoch수 만큼 반복
        for i in range(X.shape[0]):         # 입력 데이터 개수
            x = X[i, :]                     # x: 입력값 처음부터 끝까지
            t = T[i, :]                     # t: 출력값 처음부터 끝까지

            ### 신경망 순방향 계산 (forward) ##########
            u, y = forward(x, n_output)

            ### 오차역전파 역방향 계산 (backpropagation) ##########
            dE_dw, dE_db = backpropagate(x, u, y, t)

            ### 경사하강법, 가중치 업데이트 (weight Update) #####
            update_weight(dE_dw, dE_db)

            ### 에러 계산 (calculate error) #####
            error[n] = calc_error(y, t)
        print("{} EPOCH-ERROR: {}".format(n, error[n]))

    error_graph(error)

################### 신경망 모델에 의한 예측 (Prediction) ###################
def predict(x, n_output):
    u, y = forward(x, n_output)
    return u, y

if __name__ == '__main__':
##################### 학습용 입력 데이터 및 정답 데이터 #####################
    X = np.array([[0], [0.5], [1], [1.5], [2], [2.5], [3], [3.5]])   # 입력
    def fn(x):
        y = np.sqrt(x)**3
        return y
    T = np.array([[fn(X[0])],[fn(X[1])], [fn(X[2])], [fn(X[3])], [fn(X[4])], [fn(X[5])], [fn(X[6])], [fn(X[7])]])

    train(X, T)

    ############ 테스트용 입력 데이터 입력 및 예측 ###############
    x = np.array([2.2])                   # Test용 입력 데이터
    u, y = predict(x, T.shape[1])         # Test용 입력 데이터에 의한 예측
    print("Input : {} ".format(x[0]))
    print("Prediction : 루트({})의 3승->{}".format(x[0], y[0]))
    print("Real Solution is {}".format(fn(x)))
    print("")