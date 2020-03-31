인공지능개론 Project
=================
CNN을 이용한 cifar10 classification
----------------------------------
1.모델
    -ImageNet ILSVRC challenge에서 1등을 차지한 AlexNet과 VGG Net을 사용하였다.

2.학습에 영향을 준 요소
  (1) epoch 수
      -50회, 100회, 200회 등 여러 가지의 epoch 수를 사용하여 실험을 진행
      -test data에 대한 loss와 accuracy를 보고 underfit인지 overfit인지 구분하여 학습횟수 조절
      -epoch 수를 150번으로 설정
  (2) 학습률(Learning rate)
      -국소최적해에 빠지지 않기 위해서 학습률을 처음에는 크게, 그리고 점점 작게 조절
      -이 모델에서는 epoch 50까지 0.01, epoch 100까지 0.005, epoch 150까지 0.001로 학습률 설정.
      -처음부터 학습률을 너무 낮게 사용하면 학습이 오래 걸리고 좋은 결과가 보장되지도 않음.
  (3) 최적화(Optimization)
      -기본적으로 Mini-batch gradient descent를 사용
      -keras에서 SGD를 사용, batch size를 설정하여 mini-batch 형태로 구현
      -더 빠르게 학습을 진행하기 위해 momentum을 사용, momentum 값을 0.9로 설정
  (4) Batch size
      -batch size를 32,64,128,256로 하여 각 각 실험을 진행
      -batch size가 32일 때, overfit이 덜 되고 상대적으로 높은 accuracy를 얻음
      -batch size가 256일 때, 상대적으로 높은 error rate지만 학습이 빠르게 진행됨.
  (5) loss function
      -keras의 categorical cross entropy를 사용
  
