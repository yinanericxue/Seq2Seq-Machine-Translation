# Seq2Seq-Machine-Translation

# Video & PDF
https://github.com/wangshusen/DeepLearning/blob/master/Slides/9_RNN_6.pdf

# Language Dictionary 
https://www.manythings.org/anki/

# LSTM：return sequences and return state
https://www.kaggle.com/code/kmkarakaya/lstm-output-types-return-sequences-state
https://saturncloud.io/blog/simple-lstm-training-with-returnsequencestrue-in-keras/
![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/b08478c4-8faf-4f82-9e70-de519b535162)

# Stacked RNN
![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/ffb0fb5b-a4e9-4c42-ab94-cb7f3ef9324d)
![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/83fd4cb7-13b5-4b6d-ae2a-01a39c399cac)

# Text Generation:  return_sequences=False

For every sample, the input is a sequence of one-hot or ID, but the output is only one one-hot.

Char-Level,  input: 60 x one-hot(39), output: 1 x one-hot(39)
Word-Level,  input: 18 x ID(0~2241), output: 1 x one-hot(2242)
![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/266be835-7935-4add-8544-ec22ab725be7)

The samples are generated to support this training method:
![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/bd4fc2f5-4a81-4ae9-a281-778bbeee63d0)

# Text Generation:  if return_sequences == True

For a sample, for every time (one-hot or ID) in the input sequence, there should be a one-hot in the output.
![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/07d5ec7e-90bc-4c42-9125-5d5b355eb32f)

##### Teacher Forcing for training 
https://www.cnblogs.com/dangui/p/14690919.html
![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/531f1830-d20a-4e39-a48f-9c5d266b5c1f)
![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/a39f739c-26bb-4bbe-baf5-609c6cdfc1a4)


##### Functional API:  Share Layers by multiple models
https://stackoverflow.com/questions/51032625/shared-layers-different-models
![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/ca4adc45-c9a8-4b18-8032-32e0af58df42)


Multiple Inputs and Outputs;
Shared Layers within the same model;
https://github.com/keras-team/keras/issues/12261

Multiple Models with Shared Layers;
https://saturncloud.io/blog/saving-keras-models-with-shared-layers-a-comprehensive-guide/

![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/286c09b6-e406-4390-a139-d7f6767ae79d)

##### Code: English -> Chinese, char-level, no embedding layer
https://blog.csdn.net/qq_44635691/article/details/106919244![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/8269dd6c-0620-4584-8829-cb1a4a31f8ec)




![Screen Shot 2023-12-20 at 12 52 39 PM](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/6fdbf211-77f6-4eb1-8ad3-cbd57656ee1e)


# Training Model

1 Forward Propagation: S x 18 x 67 -> S x 16 x 1450, S means the number of sample in a batch 
 

![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/76bf6c9f-a609-4898-a8ca-bb616d1c4766)

(67 + 256 +1) * 256 * 4 = 331,776 
(1450 + 256 +1) * 256 * 4 = 1747968 
(256+1)*1450 = 372650

![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/36fea877-0d64-47c7-907e-a905e5703095)

# Inference Model: Encoder, Decoder

Encoder, 1 Forward Propagation: S x 18 x 67 -> S x [ 256, 256]
Decoder, handle 1 word per call.

The weights in the LSTM and Dense are shared between training and inference.
The inputs don't contain weights.

![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/77f35d12-6418-4fe8-98a7-c781425f4f6c)


![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/f3b5fcb8-8067-4bb8-95cb-f64f05318bdc)


![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/9406495b-ddf3-4a46-adf0-f010be07ca60)


![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/20f9500f-9372-46ca-b97d-a105f5a99ae7)


##### Code: English -> French, char-level, no embedding layer
https://keras.io/examples/nlp/lstm_seq2seq/
https://medium.com/analytics-vidhya/encoder-decoder-seq2seq-models-clearly-explained-c34186fbf49b

![Screen Shot 2023-12-20 at 12 54 54 PM](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/d20a2a93-9423-4ef0-8a49-37010f9e6e48)

![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/72bfa09c-e3e8-4e0b-a920-e1c2f3b8743d)

![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/0f92115f-741a-4ac2-9b69-e400f13778fa)

(70 + 256 + 1) x 256 x 4 = 334 848 
(93 + 256 + 1) x 256 x 4 = 358 400 
(256 + 1) x 93 = 23 901

![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/3c42e10f-c208-4c13-9c12-aeaf5da547bd)

# Inference Model: Encoder, Decoder

Free-running mode
![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/113027ad-105e-4b67-b79d-6edc3a95a69e)

##### Keras Input Shape
https://keras.io/api/layers/core_layers/input/
https://datascience.stackexchange.com/questions/54159/what-does-an-input-layer-of-shape-none-or-none-12-actually-mean

![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/c3759efb-58de-4a76-b49e-0338538eb89e)
![image](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/953a79bc-2e62-492c-97c5-77c4d4f064fb)
![Screen Shot 2023-12-20 at 12 56 52 PM](https://github.com/yinanericxue/Seq2Seq-Machine-Translation/assets/102645083/1efe635c-d6a3-47ed-8c1e-5734a96e9f92)

