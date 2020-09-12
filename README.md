# CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition 
A PyTorch implementation of [Convolutional Recurrent Neural Network](https://arxiv.org/abs/1507.05717) for scene text recognition.
The author's original implementation can be found [here](https://github.com/bgshih/crnn).

A novel neural network architecture, which integrates feature extraction, sequence modeling and transcription into a unified framework, is proposed for image-based sequence recognition tasks, such as scene text recognition and OCR. Below are a few examples from prediction results:

| demo images                                                | CRNN           | CRNN(case sensitive)           |
| ---                                                        |---             | ---                            |
| <img src="./images/demo_1.png" width="300">                |   available    |  Available                     |
| <img src="./images/demo_3.png" width="300">                |   londen       |   Fonden                       |
| <img src="./images/demo_4.png" width="300" height="100">   |    future      |   Future                       |
| <img src="./images/demo_8.png" width="300" height="100">   |    grred       | Gredl                          |

## Recurrent Neural Networks
### Sequential Data
Sequential data or time-series data is any kind of data where the order matters, one thing follows another. Sequential data comes in many forms such as audio, video, text, etc. To illustrate, say you take a screenshot of the video and then you want to predict the action of the person in that video. Harly can you perform such task without knowledge of previous frames of the video. But if you take many screenshots of that person in succession, maybe you will have enough information to make a prediction.

Another example, you can break text up into a sequence of words. Say "I am Vietnamese", if you shuffer the order, it will impact directly to the original meaning. The order of each word in the sequence is really important to express the sentence's contents.

### Recurrent Neural Networks
Traditional neural networks, also known as feed-forward neural network, we assume that all inputs (and outputs) are independent of each other, information moves in only one direction, forward, from the input nodes, through the hidden nodes (if any) to the output nodes.

# anh feed forward

A feed-forward neural network are not able to use previous information to effect later ones. But Recurrent Neural Networks address this issue. They are networks with loops that carries information from one step to the next, allowing information to persist.

# anh rnn BPTT vanishing gradient LSTM biLSTM


