# iXjXk의 일반화 신경망을 만들어 보자. 오미입, 오메가 미입 공식 활용
# 활성화는 sigmoid. 정규화, 역정규화를 통해 임의의 값에 대응
# 학습 및 테스트용 데이터셋을 직접 입력하여 학습/예측 
# (주의!) list안의 숫자가 스케일이 다르면 정규화 효과가 떨어지므로, 우선 스케일을 맞출것!!! 

import numpy as np
from random import random
import matplotlib.pyplot as plt

######################### Hyper Parameter ######################
alpha = 0.5       # 학습율 (Learning Rate)
epoch = 20000     # 학습 에포크(학습 횟수)
n_hidden = 7      # 은닉층 노드수

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

# 어떤 list를 기준으로 list 범위안에 포함되는 임의의 숫자를 정규화
# def norm_data(lst, x): 
#     norm_num = (x - min(lst))/(max(lst) - min(lst)) 
#     return norm_num

# 어떤 list전체를 역정규화(denormalization)하기 위한 함수
def denorm_list(lst1, lst2):  # lst1은 정규화에 사용된 list, lst2는 역정규화하기 위한 list(예측값)
    denormalized = []
    for value in lst2:
        denormal_num = value * (max(lst1) - min(lst1)) + min(lst1)       # 역정규화 공식
        denormalized.append(denormal_num)   
    return denormalized

# 어떤 list를 기준으로 정규화된 숫자를 원래의 값으로 변환하는 함수
# def denorm_data(lst, x):  
#     denormal_num = x * (max(lst) - min(lst)) + min(lst)       # 역정규화 공식
#     return denormal_num

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
    plt.ylim(0.0, 0.05)    # 그래프의 y축 표시 범위(0.01부분을 조정하자!)
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
    u, y = forward(x, n_output)  # 학습 후, 가중치, 기준치가 정해지면 순방향으로 출력
    return u, y                  # u는 은닉층 출력, y는 출력층 출력(최종 출력)

if __name__ == '__main__':
##################### 학습용 입력 데이터 및 정답 데이터 #####################
    X = np.array([[71,26,31], [82,25,29], [90,24,28],   # 입력값: 
                  [68,40,52], [81,42,51], [90,37,48],   # (몸무게, 섭취열량/100, 활동량)
                  [72,54,67], [78,52,68], [91,55,69], 
                  [69,48,59], [79,50,61], [89,52,63]])      
    T = np.array([[40,61], [ 10,89], [-20,79],           # 정답값:
                  [50,51], [  0,97], [-50,51],           # (체증증감/10, 컨디션)
                  [60,41], [-10,88], [-80,19],        
                  [80,20], [ 20,80], [-40,59]])         

#     몸무게,  섭취열량,  활동량 -> 체중증감  컨디션       
#       71Kg  2600Kcal   31Kcal/Kg    399g    61%  
#       82Kg  2500Kcal   29Kcal/Kg    103g    89%
#       90Kg  2400Kcal   28Kcal/Kg   -198g    79%
#       68Kg  4000Kcal   52Kcal/Kg    501g    51%            
#       81Kg  4200Kcal   51Kcal/Kg      3g    97%
#       90Kg  3700Kcal   48Kcal/Kg   -498g    51%
#       72Kg  5400Kcal   67Kcal/Kg    597g    41%
#       78Kg  5200Kcal   68Kcal/Kg   -103g    88%
#       91Kg  5500Kcal   69Kcal/Kg   -802g    19%
#       69Kg  4800Kcal   59Kcal/Kg    804g    20%
#       79Kg  5000Kcal   61Kcal/Kg    201g    80%
#       89Kg  5200Kcal   63Kcal/Kg   -397g    59%

#       85Kg  4500Kcal   60Kcal/Kg     ??g    ??%

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
    test = np.array([85, 45, 60])   # Test용 입력 데이터(몸무게, 섭취열량/100, 활동량)
    n_test = norm_list(X1, test)          # Test용 데이터의 정규화
    u, y = predict(n_test, T.shape[1])    # Test용 입력 데이터에 의한 예측

    denorm_y = denorm_list(T1, y)         # 예측 출력값의 역정규화
 
    print("내 몸무게는 {}Kg 인데, {}Kcal를 먹고, {}Kcal/Kg 활동했어요."
                            .format(test[0], test[1]*100, test[2]))
    print("과연 체중은 얼마나 변하고, 또 컨디션은 어떨까요?")
    print("체중증감: {:.2f}g, 컨디션: {:.2f}%".format(denorm_y[0]*10, denorm_y[1]))
    print("")