import argparse

from acpsr.evaluation import start_evaluate
from acpsr.data.generate_combined_prescription import start_generation
import acpsr.data.reader as reader
from acpsr.data.statistic import start_statistic

from acpsr.train.train import start_train
from acpsr.train.dataset import split_dataset


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(prog='acpsr', description='acupoint speech recognition')
    # Add an argument "command" which is expected to be a string
    parser.add_argument('command', metavar='C', type=str, nargs='+', help='command')
    # Parse the command line arguments
    args = parser.parse_args()

    # Check the value of the "command" argument
    if args.command[0] == "generate":
        # Call the "start_generation" function if the "command" argument is "generate"
        sample_num = start_generation()

    elif args.command[0] == "evaluate":
        # Check the second item in the "command" argument
        if args.command[1] not in ['rule_seg', 'rnn_seg']:
            print('Error: Choose `rule_seg` or `rnn_seg` segment method to evaluate.')
            exit(1) 
        # Check the third item in the "command" argument
        if args.command[2] not in ['comb_pres', 'pres']:
            print('Error: Choose `comb_pres` or `pres` to evaluate.')
            exit(1)
        
        # Store the segment method in a variable
        seg_method = args.command[1]
        # Determine whether the combined presentation is used
        combined_pres = args.command[2] == 'comb_pres'

        # Call the "start_evaluate" function with the determined values of "combined_pres" and "seg_method"
        start_evaluate(combined_pres, seg_method)
        
    elif args.command[0] == "train":
        # Check the second item in the "command" argument
        if args.command[1] == "rnn_seg":
            # Call the "split_dataset" function
            split_dataset()
            print('Start to load data in hard drive into memory, it takes a lot of time!')
            # Call the "start_train" function
            start_train()

    elif args.command[0] == "statistic":
        # Call the "reader.statistic" function
        start_statistic()
