import tensorflow as tf
import argparse
from train_test_eval import train, evaluate
import os
import time

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_enc_len", default=512, help="Encoder input max sequence length", type=int)
  parser.add_argument("--max_dec_len", default=128, help="Decoder input max sequence length", type=int)
  parser.add_argument("--max_dec_steps", default=128, help="maximum number of words of the predicted abstract", type=int)
  parser.add_argument("--min_dec_steps", default=32, help="Minimum number of words of the predicted abstract", type=int)
  parser.add_argument("--batch_size", default=4, help="batch size", type=int)
  parser.add_argument("--beam_size", default=4, help="beam size for beam search decoding (must be equal to batch size in decode mode)", type=int)
  parser.add_argument("--vocab_size", default=50000, help="Vocabulary size", type=int)
  parser.add_argument("--embed_size", default=128, help="Words embeddings dimension", type=int)
  parser.add_argument("--enc_units", default=256, help="Encoder GRU cell units number", type=int)
  parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
  parser.add_argument("--attn_units", default=512, help="[context vector, decoder state, decoder input] feedforward result dimension - this result is used to compute the attention weights", type=int)
  parser.add_argument("--learning_rate", default=0.15, help="Learning rate", type=float)
  parser.add_argument("--adagrad_init_acc", default=0.1, help="Adagrad optimizer initial accumulator value. Please refer to the Adagrad optimizer API documentation on tensorflow site for more details.", type=float)
  parser.add_argument("--max_grad_norm",default=0.8, help="Gradient norm above which gradients must be clipped", type=float)
  parser.add_argument("--checkpoints_save_steps", default=1000, help="Save checkpoints every N steps", type=int)
  parser.add_argument("--max_checkpoints", default=10, help="Maximum number of checkpoints to keep. Olders ones will be removed", type=int)
  parser.add_argument("--max_steps", default=50000, help="Max number of iterations", type=int)
  parser.add_argument("--num_to_test", default=100, help="Number of examples to test", type=int)
  parser.add_argument("--max_num_to_eval", default=100, help="Max number of examples to evaluate", type=int)
  parser.add_argument("--mode", help="training, eval or test options", default="", type=str)
  parser.add_argument("--model_name", help="Name of a specific model", default="", type=str)
  parser.add_argument("--checkpoint_dir", help="Checkpoint directory", default="./checkpoint/", type=str)
  parser.add_argument("--test_save_dir", help="Directory in which we store the decoding results", default="./test/", type=str)
  parser.add_argument("--data_dir",  help="Data Folder", default="", type=str)
  parser.add_argument("--vocab_path", help="Vocab path", default="", type=str)
  parser.add_argument("--log_dir", help="Directory in which to redirect console outputs", default="./log/", type=str)


  args = parser.parse_args()
  params = vars(args)
  print(params)

  assert params["mode"], "mode is required and must be train or eval"
  assert params["mode"] in ["train", "eval"], "The mode must be train or eval"
  assert (not params["model_name"]) or (params["model_name"].upper() in ["LSTM_LSTM", "LSTM_GRU", "GRU_LSTM", "GRU_GRU"]), "The model name must be empty or one of these: LSTM_LSTM, LSTM_GRU, GRU_LSTM or GRU_GRU"
  assert os.path.exists(params["data_dir"]), "data_dir doesn't exist"
  assert os.path.isfile(params["vocab_path"]), "vocab_path doesn't exist"

  t0 = time.time()
  if params["mode"] == "train":
    train( params)
  elif params["mode"] == "eval":
    evaluate(params)
  t1 = time.time()
  total = t1-t0
  print(f"Time Taken: {0}", total)
  
  
if __name__ =="__main__":
  main()