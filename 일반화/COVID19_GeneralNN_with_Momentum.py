# iXjXk의 일반화 신경망을 만들어 보자. 오미입, 오메가 미입 공식 활용
# 활성화 함수는 sigmoid와 tanh 선택가능. 출력값 범위 주의할 것  
# 입력부분에 1이 추가된 bias생략 모델임

# Corona Virus 감염여부 예측
#  발열, 미후각, 기침, 가슴통증    감기/독감   코로나       
#   1      0      0      1            0         1
#   1      0      0      0            0.5      0.5
#   0      0      1      1            1         0
#   0      1      0      0            0         0
#   1      1      0      0            0         1
#   0      1      0      1            0        0.5
#   0      0      1      0            1         0

import random
import numpy as np

random.seed(0)  # 동일한 난수 셋을 생성하기 위한 방법, 뒤의 숫자는 중요하지 않음

############### 학습용 데이터셋 준비 (입력 데이터, 정답 데이터) ###########
data = [
    [[1,0,0,1], [0, 1]],
    [[1,0,0,0], [0.5, 0.5]],
    [[0,0,1,1], [1, 0]],
    [[0,1,0,0], [0, 0]],
    [[1,1,0,0], [0, 1]],
    [[0,1,0,1], [0, 0.5]],
    [[0,0,1,0], [1, 0]]
]

##### Hyper Parameter: 학습회수(epoch), 학습률(lr), 모멘텀 계수(mc) 설정 #####
epoch = 10000
lr = 0.1
mc = 0.3

##################  활설화 함수 정의 ##################
# 활성화 함수: 시그모이드 (sigmoid), 하이퍼 탄젠트(tanh)
# 함수 원래값과 미분값을 선택가능
def sigmoid(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def tanh(x, derivative=False):
    if (derivative == True):
        return (1 - x) * (1 + x)
    return np.tanh(x)

################### 가중치 배열 설정 (학습하면서 업데이트) ############
def weight_list(i, j, fill=0.0):
    wt = []
    for i in range(i):
        wt.append([fill] * j)
    return wt

################### 신경망 모델 구축 (Building a NN Model) ############
class NeuralNetwork:
    def __init__(self, num_x, num_h, num_y, bias=1):

        # 입력층 노드(num_x), 은닉층 노드(num_h), 출력층 노드(num_y), 바이어스
        self.num_x = num_x + bias  # 바이어스 생략모델 (입력에 1 추가)
        self.num_h = num_h
        self.num_y = num_y

        # 활성화 함수 초기값
        self.activation_input = [1.0] * self.num_x
        self.activation_hidden = [1.0] * self.num_h
        self.activation_out = [1.0] * self.num_y

        # 은닉층 가중치 초기값
        self.weight_in = weight_list(self.num_x, self.num_h)
        for i in range(self.num_x):
            for j in range(self.num_h):
                self.weight_in[i][j] = random.random()

        # 출력층 가중치 초기값
        self.weight_out = weight_list(self.num_h, self.num_y)
        for j in range(self.num_h):
            for k in range(self.num_y):
                self.weight_out[j][k] = random.random()

        # 모멘텀 SGD를 위한 이전 가중치 초깃값
        self.gradient_in = weight_list(self.num_x, self.num_h)
        self.gradient_out = weight_list(self.num_h, self.num_y)

    ######### 업데이트 함수 (테스트용) #########
    def update(self, inputs):

        # 입력 레이어의 활성화 함수
        for i in range(self.num_x - 1):
            self.activation_input[i] = inputs[i]

        # 은닉층의 활성화 함수
        for j in range(self.num_h):
            sum = 0.0
            for i in range(self.num_x):
                sum = sum + self.activation_input[i] * self.weight_in[i][j]
            # 시그모이드와 tanh 중에서 활성화 함수 선택
            self.activation_hidden[j] = sigmoid(sum, False)

        # 출력층의 활성화 함수
        for k in range(self.num_y):
            sum = 0.0
            for j in range(self.num_h):
                sum = sum + self.activation_hidden[j] * self.weight_out[j][k]
            # 시그모이드와 tanh 중에서 활성화 함수 선택
            self.activation_out[k] = sigmoid(sum, False)

        return self.activation_out[:]
    
    ############# 오차 역전파(Back Propagation) 계산 ##############
    def backPropagate(self, targets):

        # 출력층 델타 (델타="오미입"중 오미부분) 계산
        output_deltas = [0.0] * self.num_y
        for k in range(self.num_y):
            error = targets[k] - self.activation_out[k]
            # 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
            output_deltas[k] = sigmoid(self.activation_out[k], True) * error # 오차x미분

        # 은닉층 델타 (델타="오미입"중 오미부분) 계산
        hidden_deltas = [0.0] * self.num_h
        for j in range(self.num_h):
            error = 0.0
            for k in range(self.num_y):
                error = error + output_deltas[k] * self.weight_out[j][k]  # "오메가" 부분
                # 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
            hidden_deltas[j] = sigmoid(self.activation_hidden[j], True) * error # 오차x미분 

        # 출력층 가중치 업데이트 (경사하강법+모멘텀)
        for j in range(self.num_h):
            for k in range(self.num_y):
                gradient = output_deltas[k] * self.activation_hidden[j]  # 오미입
                v = mc * self.gradient_out[j][k] - lr * gradient
                self.weight_out[j][k] += v
                self.gradient_out[j][k] = gradient

        # 은닉층 가중치 업데이트 (경사하강법+모멘텀)
        for i in range(self.num_x):
            for j in range(self.num_h):
                gradient = hidden_deltas[j] * self.activation_input[i] # 오메가미입
                v = mc*self.gradient_in[i][j] - lr * gradient
                self.weight_in[i][j] += v
                self.gradient_in[i][j] = gradient

        # 오차의 계산(평균 제곱 오차)
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.activation_out[k]) ** 2
        return error

    ############# 학습 실행 ###########
    def train(self, patterns):
        for i in range(epoch ) :
            error =  0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets)
            if i % 500 == 0:
                print('error: %-.5f' % error)
    ############ 예측값 출력 ##########
    def predict(self, patterns):
        for p in patterns:
            print('Input: %s, Predict: %s' % (p[0], self.update(p[0])))

if __name__ == '__main__':

    # n = NeuralNetwork(입력개수, 은닉층 노드수, 출력 개수)
    n = NeuralNetwork(4, 4, 2)  # NeuralNetwork 클래스의 인스턴스: n
    # 학습 실행
    n.train(data)  # 학습용 테이터로 학습
    # 예측값 출력
    n.predict(data) # 학습용 데이터로 예측, 정답값과 비교해 볼 것

    ######### 테스트 (발열과 기침에 대해 코로나 여부 예측) ########
    test_data = [1, 0, 1, 0]
    rst = n.update(test_data)
    print("")
    print('Test Input(Fever, Cough): %s, Predict(Cold? Corona?): %s' % (test_data, rst))