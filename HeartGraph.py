# 하트 방정식을 그래프로 그려보자. 
# x^2 + (y-sqrt(abs(x)))^2 = 2
# 윗식을 변형하여 극좌표 함수로 만들고 프로그래밍하자.
# 그래프 상에서는 옆으로 좀 퍼져있으니, 폭을 조종하여 예쁜 하트를 만들어 보자! 

from matplotlib import pyplot as plt
from math import pi, sin, cos, sqrt

def draw_graph(x, y, title, color, linewidth):
  plt.title(title)
  plt.plot(x, y, color=color, linewidth=linewidth)
  plt.show()

# frange()는 range()함수의 부동소수점수 버젼
def frange(start, final, increment=0.01):
  numbers = []

  while start < final:
    numbers.append(start)
    start = start + increment
  return numbers

def draw_heart():
  intervals = frange(0, 2 * pi)
  x = []
  y = []

  for t in intervals:  # 아래의 두줄이 하트 모양을 나타내는 방정식
    x.append(sqrt(2)*sin(t))
    y.append(sqrt(2)*(cos(t)+sqrt(abs(sin(t)))))

  draw_graph(x, y, title='HEART', color='red', linewidth=5)

if __name__ == '__main__':
  try:
    draw_heart()
  except KeyboardInterrupt:
    pass