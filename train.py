#!/usr/bin/env python

"""

This is a traiing script for sequence to sequence learning.

"""

# TODO train and validation output frequency
# TODO logovani do souboru
# TODO pripravit ini soubor pro spusteni modelu
# TODO better handling parsing INI errors

import sys
import os
import codecs
from shutil import copyfile

import tensorflow as tf

from utils import print_header, log
from configuration import Configuration
from learning_utils import training_loop, initialize_tf
from dataset import Dataset
from config_generator import save_configuration

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: train.py <ini_file>"
        exit(1)

    config = Configuration()
    config.add_argument('name', str)
    config.add_argument('random_seed', int, required=False)
    config.add_argument('output', basestring)
    config.add_argument('epochs', int, cond=lambda x: x >= 0)
    config.add_argument('trainer')
    config.add_argument('encoders', list, cond=lambda l: len(l) > 0)
    config.add_argument('decoder')
    config.add_argument('batch_size', int, cond=lambda x: x > 0)
    config.add_argument('train_dataset', Dataset)
    config.add_argument('val_dataset', Dataset)
    config.add_argument('postprocess')
    config.add_argument('evaluation', cond=list)
    config.add_argument('runner')
    config.add_argument('test_datasets', list, required=False, default=[])
    config.add_argument('initial_variables', str, required=False, default=[])
    config.add_argument('validation_period', int, required=False, default=500)

    try:
        ini_file = sys.argv[1]
        log("Loading ini file: \"{}\"".format(ini_file), color='blue')
        config_f = codecs.open(ini_file, 'r', 'utf-8')
        args = config.load_file(config_f)
        log("ini file loded.", color='blue')
    except Exception as exc:
        log(exc.message, color='red')
        exit(1)

    print ""
    print_header(args.name)

    if args.random_seed is not None:
        tf.set_random_seed(args.random_seed)

    try:
        os.mkdir(args.output)
    except:
        log("Experiment directory \"{}\" already exists".format(args.output), color='red')
        exit(1)

    copyfile(ini_file, args.output+"/experiment.ini")
    os.system("git log -1 --format=%H > {}/git_commit".format(args.output))
    os.system("git --no-pager diff --color=always > {}/git_diff".format(args.output))

    run_configuration = {
        'encoders': args.encoders,
        'decoder': args.decoder,
        'runner': args.runner,
        'evaluation': args.evaluation,
        'initial_variables': args.output+"/variables.data"
    }
    save_configuration(run_configuration, args.output)


    sess, saver = initialize_tf(args.initial_variables)
    training_loop(sess, saver, args.epochs, args.trainer, args.encoders + [args.decoder], args.decoder,
                  args.batch_size, args.train_dataset, args.val_dataset,
                  args.output, args.evaluation, args.runner,
                  test_datasets=args.test_datasets, validation_period=args.validation_period)
