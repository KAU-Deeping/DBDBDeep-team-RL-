'''
##Tensorboard
###abstract
텐서플로우 그래프를 시각화하고, 그래프의 실행에 관한 정량적인 지표(quantitative matrix)들을 그래프화(plot) 하고, 학습과정에서 전달되는 이미지와 같은 추가적인 데이터를 보여주기 위해 텐서보드를 사용

텐서보드는 텐서플로우를 실행할때 생성할 수 있는 텐서플로우 event file(summary data)을 읽으면서 동작한다.

요약데이터를 수집하고 싶은 텐서플로우 그래프를 생성
summary ops를 이용해서 어떤 노드를 annotate 하고 싶은지를 결정 (learning rate or loss) tf.summary.scalar ops 사용헤서 scalar 값 수집<br>
distribution, gradient weight들의 분포를 시각화할때 tf.summary.histogram ops 사용<br><br>

summary node들은 그래프에 붙어서 주변적으로 존재한다.
summary 데이터를 생성하기 위해서는 그 노드들을 run 시켜줘야만 한다.
이때 tf.summary.merge_all 이용해서 하나의 single op로 묶고 이를 이용해서 요약 데이터를 생성한다. 그 다음에 통합된 summary ops를 싱행하면 serialized 된 summary protobuf를 생성한다. protobuf는 각 step에 summary data를 가지고 있다. 이걸 디스크에 쓰기 위해서 summary protobuf를 tf.summary.FileWritier에 넘겨준다. 

	graph 생성, tf.summary.scalar ops사용
	변화를 알고 싶은 것 정해서 tag 붙이기 (loss ㄲ)
  	distribution -> tf.summary.histogram

'''