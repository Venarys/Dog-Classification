# Dog-Classification
> This is a project which aims to be a practice of 
image processing for myself.

The main idea of this project is to classficate the breed of a dog rely on a picture of a perticular breed dog. Make use of the ResNet-50 pre-trained model and do fining on that. 

I freeze the former layers to ruduces the cost of training and try to give play to the pre-trained model. but I found the time of training descend a little and the loss ascend a bit. I think it is a bit unworthy. Full-parameters training is good.

It takes less than half an hour to train the model in my R5-cpu.

The code is helped by Qwen, and the run_old.py used the PaddlePaddle framework, but I and Qwen can not run the code, So I turn to the Pytorch, more widely used model.