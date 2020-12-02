# Tutorial on Knowledge Distillation with Keras
This tutorial aims to introduce the concept of knowledge distillation and its application in medical image processing. 

# What is knowledge distillation:
Knowledge distillation refers to any type of training which transfers knowledge from a complex or deep model to a simple or compact model. It was introduced first by Bucilua et al. [1] as a model compression method to compress a large ensemble model into a more compact model. The method was proven to lower the computation time and  memory requirements, while maintaining the performance. Later, this concept was further used in the “Teacher-Student” setting, where the student model is trained to mimic the knowledge acquired by the teacher model such that it produces the same predictions.
Knowledge distillation can be used as a method to handle the lack of data to train a deep neural network. It aims to transfer the knowledge from the teacher model trained on the source data to facilitate learning the student model on target task. This framework was initially designed to transfer the knowledge between the models trained on the same data. Recently, some studies introduced the cross-modal knowledge distillation to transfer the learned representations from one source of data (side modality) to another (main modality)[2].

![image](https://user-images.githubusercontent.com/41435220/100928868-150a0500-34e7-11eb-92b4-f571634048e4.png)

[1] Bucilua, C., Caruana, R., Niculescu-Mizil, A.: Model compression. In: Proceedingsof the 12th ACM SIGKDD international conference on Knowledge discovery anddata mining. pp. 535–541 (2006)

[2]  Gupta, S., Hoffman, J., Malik, J.: Cross modal distillation for supervision transfer.In: Proceedings of the IEEE conference on computer vision and pattern recognition.pp. 2827–2836 (2016)
