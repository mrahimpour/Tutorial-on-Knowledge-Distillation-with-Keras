# Tutorial on Knowledge Distillation with Keras
This tutorial aims to introduce the concept of knowledge distillation and its application in medical image processing. 

# What is knowledge distillation:
Knowledge distillation refers to any type of training which transfers knowledge from a complex or deep model to a simple or compact model. It was introduced first by Bucilua et al. [1] as a model compression method to compress a large ensemble model into a more compact model. The method was proven to lower the computation time and  memory requirements, while maintaining the performance. Later, this concept was further used in the “Teacher-Student” setting, where the student model is trained to mimic the knowledge acquired by the teacher model such that it produces the same predictions. This is accomplished by using the class probabilities produced by softmax layer of the teacher model as the target values to train the student model by minimizing the cross-entropy objective function.

However, small values of teacher's class probabilities have only a small contribution to the cross-entropy function. While these small values can represent valuable information about the similarity structure in the data~\cite{hinton2015distilling}, they will have limited impact on the training of the student network. To overcome this disadvantage, Be and Caruana~\cite{ba2014deep} proposed to use logit values as targets instead of the final prediction values of the teacher model. Logit values are defined as the pre-softmax activations and therefore contain logarithmic relationship between prediction probabilities. 
%As such, the student model could learn a richer representation from the teacher model. 
Instead of using logit values, Hinton et al.~\cite{hinton2015distilling} proposed the concept of temperature to soften the target values of the teacher and to provide a better representation of smaller probabilities in the output values. In addition to optimizing the student model with the ground truth labels, their proposed objective function penalizes the student model based on the softened version of teacher output as follows: 
%Their proposed objective function to train the student model is as follows:

\begin{equation}
    L_{KD} = (1-\alpha)H(y, y^{S})+\alpha T^{2}H(\sigma(z^{T}/T), \sigma(z^{S}/T))
    \label{eq:1}
\end{equation}
Knowledge distillation can be used as a method to handle the lack of data to train a deep neural network. It aims to transfer the knowledge from the teacher model trained on the source data to facilitate learning the student model on target task. This framework was initially designed to transfer the knowledge between the models trained on the same data. Recently, some studies introduced the cross-modal knowledge distillation to transfer the learned representations from one source of data (side modality) to another (main modality)[2].

![image](https://user-images.githubusercontent.com/41435220/100928868-150a0500-34e7-11eb-92b4-f571634048e4.png)

[1] Bucilua, C., Caruana, R., Niculescu-Mizil, A.: Model compression. In: Proceedingsof the 12th ACM SIGKDD international conference on Knowledge discovery anddata mining. pp. 535–541 (2006)

[2]  Gupta, S., Hoffman, J., Malik, J.: Cross modal distillation for supervision transfer.In: Proceedings of the IEEE conference on computer vision and pattern recognition.pp. 2827–2836 (2016)
