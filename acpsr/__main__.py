import argparse

from acpsr.evaluation import start_evaluate
from acpsr.data.generate_combined_prescription import start_generation
import acpsr.data.reader as reader

from acpsr.train.train import start_train
from acpsr.train.dataset import split_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'acpsr',
                    description = 'acupoint speech recognition')
    parser.add_argument('command', metavar='C', type=str, nargs='+',
                        help='command')
    args = parser.parse_args()

    if args.command[0] == "generate":
        sample_num = start_generation()

    if args.command[0] == "evaluate":
        if args.command[1] != "rule_seg" and args.command[1] != 'rnn_seg':
            print('choose `rule_seg` or `rnn_seg` to evaluate.')
            exit(1) 
        start_evaluate(args.command[1])
        


    
    if args.command[0] == "train":
        if args.command[1] == "rnn_seg":
            split_dataset()
            print('start to load data in hard drive into the memory, it takes a lot of time!')
            start_train()


