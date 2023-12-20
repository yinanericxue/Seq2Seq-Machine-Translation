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




