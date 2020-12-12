# Tutorial on Knowledge Distillation with Keras
This tutorial aims to introduce the concept of knowledge distillation and its application in medical image processing. 

# What is knowledge distillation:
Knowledge distillation refers to any type of training which transfers knowledge from a cumbersome model to a simple or compact model. It was introduced first by Bucilua et al. [1] as a model compression method to compress a large ensemble model into a more compact one. The method was proven to lower the computation time and  memory requirements, while maintaining the performance. Later, this concept was further used in the “Teacher-Student” setting, where the student model is trained to mimic the knowledge acquired by the teacher model such that it produces the same predictions. This is accomplished by using the class probabilities produced by softmax layer of the teacher model as the target values to train the student model by minimizing the cross-entropy objective function.

However, small values of teacher's class probabilities have only a small contribution to the cross-entropy function. While these small values can represent valuable information about the similarity structure in the data, they will have limited impact on the training of the student network. To overcome this disadvantage, Be and Caruana [3] proposed to use logit values as targets instead of the final prediction values of the teacher model. Logit values are defined as the pre-softmax activations and therefore contain logarithmic relationship between prediction probabilities. 
Instead of using logit values, Hinton et al. [2] proposed the concept of temperature to soften the target values of the teacher and to provide a better representation of smaller probabilities in the output values. In addition to optimizing the student model with the ground truth labels, their proposed objective function penalizes the student model based on the softened version of teacher output as follows: 

![CodeCogsEqn](https://user-images.githubusercontent.com/41435220/101779817-07134000-3af6-11eb-8afd-dae7d28a69d7.gif)

Where ![CodeCogsEqn](https://user-images.githubusercontent.com/41435220/101780336-b3edbd00-3af6-11eb-99ac-5a1e4df317b8.gif) is the prediction of student model, ![CodeCogsEqn](https://user-images.githubusercontent.com/41435220/101780503-e8fa0f80-3af6-11eb-8398-4a67f19c4a81.gif) is the label representing the ground truth and H is the cross-entropy function. ![CodeCogsEqn](https://user-images.githubusercontent.com/41435220/101780662-18a91780-3af7-11eb-9c25-adbbde068a3f.gif) and ![CodeCogsEqn](https://user-images.githubusercontent.com/41435220/101926687-cf7ac580-3bd3-11eb-9711-e9a1d8a8e7cd.gif) refer to the logit values produced by teacher and student model respectively while ![CodeCogsEqn](https://user-images.githubusercontent.com/41435220/101926781-eb7e6700-3bd3-11eb-95dd-57ffef2ac984.gif) is the temperature parameter and ![CodeCogsEqn](https://user-images.githubusercontent.com/41435220/101926858-0bae2600-3bd4-11eb-80b6-91ad5f15ea59.gif) is the softmax function and ![CodeCogsEqn](https://user-images.githubusercontent.com/41435220/101928455-e3272b80-3bd5-11eb-8f6d-9c881f0d90c1.gif) is the parameter that balance the effect of hard labels provided by reference ground truth and teacher's soft labels. 

<img src="https://user-images.githubusercontent.com/41435220/101962857-77f94b80-3c0d-11eb-8c68-bd268b205baa.jpg" width="500" height="250" alt="]]">


# Implementation:
In order to setup a “Teacher-Student” framework, we need to create two separate CNNs for teacher and student models. A CNN with any arbitrary architecture can be used as the teacher and student models. It is contrary to the transfer learning approach where both models must have the same architecture to be able to copy the weights of the pre-trained model to the new model. The following steps are required to implement the knowledge distillation:

1. Train a teacher model
2. Create a student model and train it by knowledge distillation (KD) loss 

   2.1. Prepare the data
  
   2.2. Create the student model
   
   2.3. Define the KD loss function
  
   2.4. Compile and train the student-teacher model
  
   2.5. Evaluate the student model on the test dataset
  
3. Train a student from the scratch as a baseline model for comparison

**Train a teacher model:**

Initially, we create a U-Net as a teacher model. Any CNN model can be used to define the models.

    teacher_model = create_unet_like_model()
    
    objective_function = K.binary_crossentropy
    metrics_=[loss.binary_crossentropy, loss.binary_dice]
    
    teacher_model.compile(loss = objective_function,
                          optimizer = SGD(lr=initial_learning_rate, momentum=0.9, nesterov=True) if optimizer == "SGD" else Adam(lr=initial_learning_rate),
                          metrics = metrics_)
    teacher_model.fit(training_x, training_y)

**Preparing the dataset:**

The data needs to be divided into training and testing datasets, then they should be normalized. 

    def intensityNormalization(I):
         min = np.min(I, axis=(0, 1, 2))
         max = np.max(I, axis=(0, 1, 2))
         I = (I - min) / (max - min)
         return I

**Create the student model:**

    student_model = create_unet_like_model()

**Define the KD Loss:**

The distillation loss function is a weighted combination of soft dice loss between the prediction of student model ('output_student') and ground truth and the cross-entopy loss between the softened output of student model ('soft_preds') and teacher model. The loss fuunction for the distillation framework is defined as a dictionary as follows:

    loss_function = {'output_student':soft_dice, 'soft_preds':KD}
                
Where KD is defined as:                

    import keras.backend as K
    
    def KD(y_true, y_pred):
         
      d = K.int_shape(y_pred)[-1]
      ld = y_pred[...,:K.cast(d/2, 'int32')]
      rd = y_pred[...,K.cast(d/2, 'int32'):] 
      kd_loss = (α *(T*T))*K.binary_crossentropy(ld, rd)

      return kd_loss
        
**Define the student-teacher model**

    temperature = 5

    logit_student = student_model.get_layer('activation_out_student').output 
    hard_out_student = Activation('softmax')(logit_student)
   
    soft_out_student  = Lambda(lambda x: x/temperature)(logit_student)
    soft_prob_student = Activation('softmax')(soft_out_student)

    logit_teacher = teacher_model.get_layer('activation_out_teacher').output 
    soft_out_teacher = Lambda(lambda x: x/temperature)(logit_teacher)
    soft_prob_teacher = Activation('softmax')(soft_out_teacher)

    concat_out_layer_student_teacher = Concatenate(axis = -1, name = 'soft_outputss')([soft_prob_teacher, soft_prob_student])

    student_teacher_model = Model(student_model.input, [student_model.output, concat_last_layer_student_teacher])
    
    metrics_=[loss.binary_crossentropy, loss.binary_dice]
    metrics_dict = {'output_student': metrics_,'soft_preds': [])}

    student_teacher_model.compile(loss = loss_function,
                                  optimizer = SGD(lr=initial_learning_rate, momentum=0.9, nesterov=True) if optimizer == "SGD" else Adam(lr=initial_learning_rate),
                                  metrics = metrics_dict)



# Acknowledgements
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 764458 (https://www.hybrid2020.eu/).


# References:
[1] Bucilua, C., Caruana, R., Niculescu-Mizil, A.: Model compression. In: Proceedingsof the 12th ACM SIGKDD international conference on Knowledge discovery anddata mining. pp. 535–541 (2006)

[2] Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network.arXiv preprint arXiv:1503.02531 (2015)

[3]  Ba, J., Caruana, R.: Do deep nets really need to be deep? In: Advances in neuralinformation processing systems. pp. 2654–2662 (2014)

Gupta, S., Hoffman, J., Malik, J.: Cross modal distillation for supervision transfer.In: Proceedings of the IEEE conference on computer vision and pattern recognition.pp. 2827–2836 (2016)



