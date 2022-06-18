import tensorflow as tf
from model import PGN
from training_helper import train_model
from test_helper import beam_decode
from batcher import batcher, Vocab, Data_Helper
from tqdm import tqdm
from rouge import Rouge
import pprint
from os import path

def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"
    
    cellTypes = ["LSTM", "GRU"]
    for e in cellTypes:
        for d in cellTypes:
            model_name = e + "_" + d
            if (not params["model_name"]) or (model_name.upper() == params["model_name"].upper()):
                print("Training", model_name)
                model = PGN(e, d, params)

                print("Creating the vocab ...")
                vocab = Vocab(params["vocab_path"], params["vocab_size"])

                print("Creating the batcher ...")
                b = batcher(params["data_dir"], vocab, params)

                print("Creating the checkpoint manager")
                checkpoint_dir = "{0}/{1}".format(params["checkpoint_dir"], model_name)
                ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
                ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=params["max_checkpoints"])

                ckpt.restore(ckpt_manager.latest_checkpoint)
                if ckpt_manager.latest_checkpoint:
                    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
                else:
                    print("Initializing from scratch.")

                print("Starting the training ...")
                train_model(model, b, params, ckpt, ckpt_manager, "{0}/{1}.txt".format(params["log_dir"], model_name))
                model = None


def predict(encoder, decoder, params):
    model_name = encoder + "_" + decoder
    
    print("Evaluating", model_name)
    model = PGN(encoder, decoder, params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the batcher ...")
    b = batcher(params["data_dir"], vocab, params)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{0}/{1}".format(params["checkpoint_dir"], model_name)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=params["max_checkpoints"])
    
    ckpt_path = ckpt_manager.latest_checkpoint
    ckpt.restore(ckpt_path)
    print("Model restored from", ckpt_path)

    for batch in b:
        yield beam_decode(model, batch, vocab, params)


def evaluate(params):
    assert params["mode"].lower() == "eval", "change training mode to 'eval'"
    assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
    
    cellTypes = ["LSTM", "GRU"]
    for e in cellTypes:
        for d in cellTypes:
            model_name = e + "_" + d
            if not params["model_name"] or model_name.upper() == params["model_name"].upper():
                gen = predict(e, d, params)
                reals = []
                preds = []
                with tqdm(total=params["max_num_to_eval"],position=0, leave=True) as pbar:
                    for i in range(params["max_num_to_eval"]):
                        trial = next(gen)
                        reals.append(trial.real_abstract)
                        preds.append(trial.abstract)
                        if params["results_save_dir"] != None and params["results_save_dir"] != "":
                            with open(params["results_save_dir"]+"/" + model_name + "_" + str(i) + ".txt", "w") as f:
                                f.write("Article:\n")
                                f.write(trial.text)
                                f.write("\n\nReal Abstract:\n")
                                f.write(trial.real_abstract)
                                f.write("\n\nPredicted Abstract:\n")
                                f.write(trial.abstract)
                        pbar.update(1)
                r=Rouge()
                scores = r.get_scores(preds, reals, avg=True)
                print("\n\n")
                pprint.pprint(scores)