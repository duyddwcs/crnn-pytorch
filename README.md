# CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition 
A PyTorch implementation of [Convolutional Recurrent Neural Network](https://arxiv.org/abs/1507.05717) for scene text recognition.
The author's original implementation can be found [here](https://github.com/bgshih/crnn).

A novel neural network architecture, which integrates feature extraction, sequence modeling and transcription into a unified framework, is proposed for image-based sequence recognition tasks, such as scene text recognition and OCR.

## Recurrent Neural Networks
### Sequential Data
Sequential data or time-series data is any kind of data where the order matters, one thing follows another. Sequential data comes in many forms such as audio, video, text, etc. To illustrate, say you take a screenshot of the video and then you want to predict the action of the person in that video. Hardly can you perform such task without knowledge of previous frames of the video. But if you take many screenshots of that person in succession, you may have enough information to make a prediction.

Another example, you can break text up into a sequence of words. Say "I am Vietnamese", if you shuffer the order, it will impact directly to the original meaning. The order of each word in the sequence is crucial to express the sentence's contents.

### Recurrent Neural Networks
In traditional neural networks, also known as feed-forward neural network, we assume that all inputs (and outputs) are independent of each other, information moves in only one direction, forward, from the input nodes, through the hidden nodes (if any) to the output nodes.

 <img src="./images/neural_net.png"  width="800">

A feed-forward neural network are not able to use previous information to effect later ones. But Recurrent Neural Networks address this issue. They are networks with loops that carries information from one step to the next, allowing information to persist.

<img src="./images/rnn_forward.png">

- <img src="https://render.githubusercontent.com/render/math?math=x_t"> is the input at time step t.
- <img src="https://render.githubusercontent.com/render/math?math=a_t"> is the hidden state at time step t. <img src="https://render.githubusercontent.com/render/math?math=a_t"> is calculated based on the previous hidden state and the input at the current step: <img src="https://render.githubusercontent.com/render/math?math=s_t=f(Ux_t,Wa_{t-1})">. The function f usually is a nonlinearity such as tanh or ReLU. The hidden state serve as memory container of the network. It capture information about what happened in the previous time steps. 
- <img src="https://render.githubusercontent.com/render/math?math=y_t"> is the output at step t. The output at step <img src="https://render.githubusercontent.com/render/math?math=\hat{y_t}"> is calculated solely based on the memory at time t. 

Unlike a traditional deep neural network, which uses different parameters at each layer, a RNN shares the same parameters (U, V, W above) across all steps. This reflects the fact that we are performing the same task at each step, just with different inputs. This greatly reduces the total number of parameters we need to learn.

The pros and cons of a typical RNN architecture:
|Advantages|Drawbacks|
|---|---|
|- Possibility of processing input of any length|        - Computation being slow|
|- Model size not increasing with size of input |        - Difficulty of accessing information from a long time ago|
|- Computation takes into account historical information|- Cannot consider any future input for the current state|
|- Weights are shared across time||

### Different types of RNN

<img src="./images/RNN_type.jpg">

#### One to One
One to One RNN (<img src="https://render.githubusercontent.com/render/math?math=T_x">=<img src="https://render.githubusercontent.com/render/math?math=T_y">=1) is the most basic and traditional type of Neural Network giving a single output for a single input where they are independent of previous information.

Ex: Image classification.

#### One to Many
One to Many (<img src="https://render.githubusercontent.com/render/math?math=T_x">=1, <img src="https://render.githubusercontent.com/render/math?math=T_y">>1) is a kind of RNN architecture is applied in situations that give multiple output for a single input.

Ex: Image captioning, Music generation.

#### Many to One
Many to One (<img src="https://render.githubusercontent.com/render/math?math=T_x">>1, <img src="https://render.githubusercontent.com/render/math?math=T_y">=1) is a kind of RNN architecture is applied in situations when multiple inputs are required for a single output.

Ex: Sentiment classification, Video regconition.

#### Many to Many
Many to Many is a kind of RNN architecture takes multiple input and gives multiple output.
- (<img src="https://render.githubusercontent.com/render/math?math=T_x">!=<img src="https://render.githubusercontent.com/render/math?math=T_y">): This is a kind of RNN architecture where input and output layers are of different size. Ex: Machine translation.
- (<img src="https://render.githubusercontent.com/render/math?math=T_x">=<img src="https://render.githubusercontent.com/render/math?math=T_y">): This is a kind of RNN architecture where input and output layers have the same size. In other words, every input having a output. Ex: Name entity recognition.

### The problem of Short-term Memory
In the training process, recurrent neral network does a forward pass and then compares the current output and the ground truth using the cross entropy error to estimate of how poorly the network is performing. We typically treat the full sequence  as one sample, so the total error is the sum of the errors at each time steps. The gradient is calculated for each time steps with respect to the U, V and W weight parameter using the chain rule of differentiation. Going back to every time steps to update the weights starting from the error is called `Backpropogate through time`.

<img src="./images/rnn_backpropagation.png">

`Backpropogate through time` is not much different from the standard backpropagation algorithm. The key difference is that we sum up the gradients for W at each time steps because the RNN architecture share the parameters across layers. Also note that we are taking the derivative of a vector function with respect to a vector, the result is a matrix (called the Jacobian matrix) whose elements are all the pointwise derivatives.

While you are using Backpropogating through time, we adjust our weight matrices with the use of a gradient. In the process, gradients are calculated by continuous multiplications of derivatives. The value of these derivatives may be so small that these continuous multiplications may cause the gradient to practically “vanish”.The earlier layers fail to learn anything as the internal weights are barely being adjusted due to extremely small gradient. And that’s the `vanishing gradient` problem.

![](./images/vanishing_grad.gif)

Because of `vanishing gradient`, RNN’s not being able to learn on earlier time steps. In other words, the network can forget what it seen in longer sequences, thus long-term dependencies being ignored during training.

On the other hand, when the derivatives  are large, we obtain an opposite effect called `exploding gradient`, which leads to instability in the network. The problem of exploding gradients can be solved by gradient clipping i.e. if gradient is larger than the threshold, scale it by dividing. 

### LSTM Network

