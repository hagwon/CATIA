# iXjXk의 일반화 신경망을 만들어 보자. 오미입, 오메가 미입 공식 활용
# 활성화는 sigmoid. 정규화, 역정규화를 통해 임의의 값에 대응
# 3개의 숫자를 연산하는 룰을 만들고 인공지능과 사람이 같이 예측해 보기 (게임)
# 본 프로그램에서 사용한 규칙: (a,b,c) --> (a+c-b-2), (b+c-a-3)
# 규칙을 사람과 인공지능 누가 먼저 발견할까?  
# 3팀으로 나누어, A:룰 정하기, B:룰 찾기, C:AI예측, 팀 대항 게임
# A팀은 룰을 만들고, 테스트용 데이터셋을 제공 후, B, C팀에게 정답을 요구
# B팀, C팀은 테스트용 데이터셋을 제외한 데이터/정답 셋을 A에게 2개씩 요구 
# B팀, C팀중 먼저 맞춘 팀이 승리, 못 맞추면 데이터/정답셋을 추가적으로 요구

import numpy as np
from random import random
import matplotlib.pyplot as plt

######################### Hyper Parameter ######################
alpha = 0.2       # 학습율 (Learning Rate)
epoch =80000      # 학습 에포크(학습 횟수), 설정된 오차값보다 작아지면 학습정지
n_hidden = 5      # 은닉층 노드수

############ 가중치 초기화 함수 (weight initialize) ############# 
wt = []           # 가중치의 빈 list (vacant array for weights)
bs = []           # bias의 빈 list (vacant array for bias)
def init_weight(n_input, n_output):
    global wt, bs # 본 함수밖에서도 사용하기 위해 전역변수화
    for i in range(n_input*n_hidden + n_hidden*n_output):
        w = np.random.rand()
        wt.append(w)
    for i in range(n_hidden + n_output):
        w = np.random.rand()
        bs.append(w)

######## 시그모이드 활성화 함수 (sigmoid activation function) ###### 
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

################ 정규화 및 역정규화를 위한 함수 부분 ################
# list전체를 정규화(Normalization)하기 위한 함수
# 학습용 list정규화: lst1와 lst2 모두 학습용 list로 세팅 
# 테스트용 list정규화: lst1은 학습용, lst2는 테스트용
def norm_list(lst1, lst2):  # lst1:기준 list(학습용list), lst2:정규화용 list
    normalized = []
    for value in lst2:
        normal_num = (value - min(lst1))/(max(lst1) - min(lst1)) # 정규화 공식
        normalized.append(normal_num)
    return normalized

# 어떤 list전체를 역정규화(denormalization)하기 위한 함수
def denorm_list(lst1, lst2):  # lst1은 정규화에 사용된 list, lst2는 역정규화하기 위한 list(예측값)
    denormalized = []
    for value in lst2:
        denormal_num = value * (max(lst1) - min(lst1)) + min(lst1)       # 역정규화 공식
        denormalized.append(denormal_num)   
    return denormalized

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
        u.append(sigmoid(sum + bs[j]))

    # 출력층 출력값
    for k in range(n_output):
        sum = 0
        for n in range(n_hidden):
            tmp = wt[n_input*n_hidden + n*n_output+k] * u[n]
            sum = sum + tmp
        y.append(sigmoid(sum+bs[n_hidden+k]))

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
                tmp = (y[n]-t[n])*(1-y[n])*y[n]*wt[n_input*n_hidden+j+n_hidden*n] 
                # 오메가미입 중에서 "오메가" 부분을 우선적으로 계산
                sum = sum + tmp
            dE_dw.append(sum*(1-u[j])*u[j]*x[i])
                # 은칙층의 가중치 기울기 공식: 오메가 x 미입

    for j in range(n_hidden):
        sum = 0
        for k in range(n_output):
            dE_dw.append((y[k]-t[k])*(1-y[k])*y[k]*u[j])  # 오미입
            # 출력층의 가중치 기울기 공식: 오미입
            tmp = (y[k]-t[k])*(1-y[k])*y[k]*wt[n_input*n_hidden+j+n_hidden*k]
            # 은닉층의 bias 기울기: 오메가미입 중 "오메가"를 우선적으로 계산
            sum = sum + tmp
        dE_db.append(sum*(1-u[j])*u[j])
            # 은닉층 bias에 대한 기울기 공식: 오메가 x 미입 (입력은 1)
    
    for i in range(n_output):
        tmp = (y[i]-t[i])*(1-y[i])*y[i]
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
        tmp = 0.5*(y[i]-t[i])**2   # 손실함수: 평균제곱오차
        err = err + tmp
    return err

def error_graph(error):    # 학습이 진행됨(epoch)에 따른 에러값 변화를 가시화
    plt.ylim(0.0, 0.02)    # 그래프의 y축 표시 범위(0.01부분을 조정하자!)
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
        if error[n] < 0.0001: # 오차값이 설정치보다 작아지면 학습을 중단
            break

    error_graph(error)

################### 신경망 모델에 의한 예측 (Prediction) ###################
def predict(x, n_output):
    u, y = forward(x, n_output)  # 학습 후, 가중치, 기준치가 정해지면 순방향으로 출력
    return u, y                  # u는 은닉층 출력, y는 출력층 출력(최종 출력)

if __name__ == '__main__':
##################### 학습용 입력 데이터 및 정답 데이터 #####################
    X = np.array([[1,2,3], [1,2,7], [2,3,7],   # 입력값: 
                  [3,4,6], [3,5,7], [3,7,8],   # 0에서 9사이 임의의 숫자
                  [5,6,7], [5,7,9], [5,8,9], 
                  [6,7,8], [6,8,9], [7,8,9]])      
    T = np.array([[-2,1],  [-6,5],  [-4,5],    # 출력값(교시값):
                  [-1,4],  [-1,6],  [0,9],     # 앞: 첫번째+두번째-세번째-2
                  [2,5],   [1,8],   [2,9],     # 뒤: 두번째+세번째-첫번째-3
                  [3,6],   [3,8],   [4,7]])         


    X1 = X.reshape(-1)  # 정규화를 위해 1차원 배열로 정렬
    T1 = T.reshape(-1)  # 정규화를 위해 1차원 배열로 정렬
    n_X = norm_list(X1, X1)     # 학습전에 데이터를 정규화
    n_T = norm_list(T1, T1)     # 학습전에 데이터를 정규화
    
    n_X = np.array(n_X)       # reshape함수를 사용하기 위해 np.array로 변환
    n_T = np.array(n_T)       # reshape함수를 사용하기 위해 np.array로 변환
    n_X = n_X.reshape(-1, X.shape[1])   # 정규화된 1차원 배열을 다시 원래대로 복구시킴
    n_T = n_T.reshape(-1, T.shape[1])   # 정규화된 1차원 배열을 다시 원래대로 복구시킴

    train(n_X, n_T)           # 입력값과 교시값으로 학습

    ############ 테스트용 입력 데이터 입력 및 예측 ###############
    test = np.array([4,5,9])              # Test용 입력 데이터
    n_test = norm_list(X1, test)          # Test용 데이터의 정규화
    u, y = predict(n_test, T.shape[1])    # Test용 입력 데이터에 의한 예측

    denorm_y = denorm_list(T1, y)         # 예측 출력값의 역정규화
 
    print("입력 숫자: {}, {}, {}".format(test[0], test[1], test[2]))
    print("예측: {:.2f}, {:.2f}".format(denorm_y[0], denorm_y[1]))
    print("정답: {:.2f}, {:.2f}".format(test[0]+test[1]-test[2]-2, test[1]+test[2]-test[0]-3))
    print("")