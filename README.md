# Text Summarization using Pointer Generators while comparing the performance of LSTM and GRU

Compare the performance of LSTM and GRU in Text Summarization using Pointer Generator Networks as discussed in https://github.com/abisee/pointer-generator

Based on code at https://github.com/steph1793/Pointer_Generator_Summarizer

## Dataset
We use the CNN-DailyMail dataset. The application reads data in the tfrecords format files.

Dataset can be created and processed based on the instructions at https://github.com/abisee/cnn-dailymail

Alternatively, pre-processed dataset can be downloaded from https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail

## Usage

Training and Evaluation will perform the actions on all four models. Most of the parameters have defaults and can be skipped.

### Training
~~~
python main.py
--max_enc_len=400
--max_dec_len=100
--max_dec_steps=120
--min_dec_steps=30
--batch_size=4
--beam_size=4
--vocab_size=50000
--embed_size=128
--enc_units=256
--dec_units=256
--attn_units=512
--learning_rate=0.15
--adagrad_init_acc=0.1
--max_grad_norm=0.8
--mode="train"
--checkpoints_save_steps=5000
--max_steps=38000
--num_to_test=5
--max_num_to_eval=100
--vocab_path="./dataset/vocab" 
--data_dir="./dataset/chunked_train" 
--checkpoint_dir="./checkpoint" 
--test_save_dir="./test/"
--log_dir="./log/"
~~~

### Evaluation
~~~
python main.py
--max_enc_len=400
--max_dec_len=100
--max_dec_steps=120
--min_dec_steps=30
--batch_size=4
--beam_size=4
--vocab_size=50000
--embed_size=128
--enc_units=256
--dec_units=256
--attn_units=512
--learning_rate=0.15
--adagrad_init_acc=0.1
--max_grad_norm=0.8
--mode="eval"
--checkpoints_save_steps=5000
--max_steps=38000
--num_to_test=5
--max_num_to_eval=100
--vocab_path="./dataset/vocab" 
--data_dir="./dataset/chunked_val" 
--checkpoint_dir="./checkpoint" 
--test_save_dir="./test/"
--log_dir="./log/"
~~~
