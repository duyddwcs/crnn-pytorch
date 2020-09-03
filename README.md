# CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition 
A PyTorch implementation of [Convolutional Recurrent Neural Network](https://arxiv.org/abs/1507.05717) for scene text recognition.
The author's original implementation can be found [here](https://github.com/bgshih/crnn).

A novel neural network architecture, which integrates feature extraction, sequence modeling and transcription into a unified framework, is proposed for image-based sequence recognition tasks, such as scene text recognition and OCR. Below are a few examples from prediction results:

| demo images                                                | VGG-BiLSTM-CTC | VGG-BiLSTM-CTC(case-sensitive) |
| ---                                                        |---             | ---                            |
| <img src="./images/demo_1.png" width="300">                |   available    |  Available                     |
| <img src="./images/demo_3.png" width="300">                |   londen       |   Fonden                       |
| <img src="./images/demo_4.png" width="300" height="100">   |    future      |   Future                       |
| <img src="./images/demo_8.png" width="300" height="100">   |    grred       | Gredl                          |
