# iXjXk의 일반화 신경망을 만들어 보자. 오미입, 오메가 미입 공식 활용
# 활성화는 sigmoid. 정규화, 역정규화를 통해 임의의 값에 대응
# 3개의 숫자를 연산하는 룰을 만들고 인공지능과 사람이 같이 예측해 보기 (게임)
# 본 프로그램: (a,b,c) --> (a+c-b), (b+c-a)
# 규칙을 사람과 인공지능 누가 먼저 발견할까요? 게임으로 진행해 보세요. 
# 이 파일은 예측용 입니다. 학습용 파일을 사용하여 학습한 후에 이 파일을 실행하세요. 

import numpy as np
from random import random
import matplotlib.pyplot as plt
import csv

n_hidden = 5      # 은닉층 노드수

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

################### 신경망 모델에 의한 예측 (Prediction) ###################
def predict(x, n_output):
    global wt, bs

    ########## 학습된 Weight와 Bias 값을 csv파일에서 로드함 ###########
    read_csv_data = []
    with open('weights.csv', newline='') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            data = []
            for k in range(len(row)):
                data.append(float(row[k]))
            read_csv_data.append(data)
    wt = read_csv_data[0]
    bs = read_csv_data[1]

    u, y = forward(x, n_output)  # 학습 후, 가중치, 기준치가 정해지면 순방향으로 출력
    return u, y                  # u는 은닉층 출력, y는 출력층 출력(최종 출력)

if __name__ == '__main__':
##################### 학습용 입력 데이터 및 정답 데이터 #####################
    X = np.array([[1,2,3], [0,1,3], [2,3,5],   # 입력값: 
                  [2,3,6], [2,5,9], [2,4,6],   # 0에서 9사이 임의의 숫자
                  [3,4,5], [6,8,9], [1,5,8], 
                  [2,5,7], [4,4,9], [5,6,7]])      
    T = np.array([[0,4],  [-2,4],  [0,6],       # 출력값(교시값):
                  [-1,7], [-2,12], [0,8],       # 앞: 첫번째+두번째-세번째
                  [2,6],  [4,11],  [-2,12],     # 뒤: 두번째+세번째-첫번째
                  [0,10], [-1,9],  [4,8]])         

    X1 = X.reshape(-1)  # 정규화를 위해 1차원 배열로 정렬
    T1 = T.reshape(-1)  # 정규화를 위해 1차원 배열로 정렬
    n_X = norm_list(X1, X1)     # 학습전에 데이터를 정규화
    n_T = norm_list(T1, T1)     # 학습전에 데이터를 정규화
    
    n_X = np.array(n_X)       # reshape함수를 사용하기 위해 np.array로 변환
    n_T = np.array(n_T)       # reshape함수를 사용하기 위해 np.array로 변환
    n_X = n_X.reshape(-1, X.shape[1])   # 정규화된 1차원 배열을 다시 원래대로 복구시킴
    n_T = n_T.reshape(-1, T.shape[1])   # 정규화된 1차원 배열을 다시 원래대로 복구시킴

    ############ 테스트용 입력 데이터 입력 및 예측 ###############
    test = np.array([3.3,5.4,5.7])              # Test용 입력 데이터
    n_test = norm_list(X1, test)          # Test용 데이터의 정규화
    u, y = predict(n_test, T.shape[1])    # Test용 입력 데이터에 의한 예측

    denorm_y = denorm_list(T1, y)         # 예측 출력값의 역정규화
    print("입력 숫자: {}, {}, {}".format(test[0], test[1], test[2]))
    print("예측: {:.2f}, {:.2f}".format(denorm_y[0], denorm_y[1]))
    print("정답: {:.2f}, {:.2f}".format(test[0]+test[1]-test[2], test[1]+test[2]-test[0]))
    print("")