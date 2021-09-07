from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
import datetime
import time
import copy
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import numpy as np
from shutil import copyfile
import torch
import torch.optim as optim
import pickle
from datasets import FTDataset, collate_pt, load_data
from models.modeling import BertConfig
from models.order import NextDxPrediction
from models import DescTokenizer
from utils.function_helpers import get_rootCode, build_tree_with_padding, print2file
from utils.evaluation import PredictionEvaluation as Evaluation

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    # Other parameters
    parser.add_argument("--task",
                        default='dx',
                        type=str,
                        # required=True,
                        help="The prediction task, such as Mortality (mort) or Length of stay (los).")
    parser.add_argument("--data_source",
                        default='mimic',
                        type=str,
                        # required=True,
                        help="the data source: mimic III (mimic) or eICU (eicu).")
    parser.add_argument("--pretrained_dir",
                        default=None,
                        type=str,
                        # required=True,
                        help="The pre_trained model directory.")
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--add_dag",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--init_nodes",
                        default=False,
                        action='store_true',
                        help="Whether to initialize graph from description.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lamda",
                        default=1.0,
                        type=float,
                        help="The ratio between two tasks.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--gpu",
                        default=0,
                        type=int,
                        help="CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--train_ratio',
                        type=float,
                        default=0.8,
                        help="the ratio of train datasets")

    parser.add_argument('--alpha',
                        type=float,
                        default=0.01,
                        help="the ratio between previous to current attention")

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info("device: {}".format(device))

    task = args.task + '_alpha_' + str(args.alpha) + '_lr_' + str(args.learning_rate) + '_bs_' + \
           str(args.train_batch_size) + '_e_' + str(args.num_train_epochs) + '_l_' + str(args.lamda) + \
           '_tr_' + str(args.train_ratio)
    output_dir = os.path.join(args.output_dir, args.data_source, task)
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'dx_prediction.log')
    buf = '{} seed:{}, gpu:{}, num_train_epochs:{}, learning_rate:{}, train_batch_size:{}, output_dir:{}'.format(
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        args.seed, args.gpu, int(args.num_train_epochs), args.learning_rate, args.train_batch_size,
        output_dir)
    print2file(buf, log_file)

    data_path = args.data_dir
    # training data files
    seqs_file = os.path.join(data_path, args.data_source, 'inputs.seqs')
    labels_file = os.path.join(data_path, args.data_source, 'labels_next_visit.label')
    labels_visit_file = os.path.join(data_path, args.data_source, 'labels_visit_cat1.label')
    # dictionary files
    dict_file = os.path.join(data_path, args.data_source, 'inputs.dict')
    tree_dir = os.path.join(data_path, args.data_source)
    class_dict_file = os.path.join(data_path, args.data_source, 'ccs_single_level.dict')
    visit_class_dict_file = os.path.join(data_path, args.data_source, 'ccs_cat1.dict')
    code2desc_file = os.path.join(data_path, args.data_source, 'code2desc.dict')

    # model configure file
    config_json = 'models/config.json'
    # config_json = 'KEMCE/model/config.json'
    copyfile(config_json, os.path.join(output_dir, 'config.json'))

    inputs = pickle.load(open(seqs_file, 'rb'))
    labels = pickle.load(open(labels_file, 'rb'))
    labels_visit = pickle.load(open(labels_visit_file, 'rb'))

    leaves_list = []
    ancestors_list = []
    masks_list = []
    for i in range(5, 0, -1):
        leaves, ancestors, masks = build_tree_with_padding(os.path.join(tree_dir, 'level' + str(i) + '.pk'))
        leaves_list.extend(leaves)
        ancestors_list.extend(ancestors)
        masks_list.extend(masks)
    leaves_list = torch.tensor(leaves_list).long().to(device)
    ancestors_list = torch.tensor(ancestors_list).long().to(device)
    masks_list = torch.tensor(masks_list).float().to(device)

    # load configure file
    config = BertConfig.from_json_file(config_json)

    config.device = device
    config.alpha = args.alpha
    config.leaves_list = leaves_list
    config.ancestors_list = ancestors_list
    config.masks_list = masks_list
    vocab = pickle.load(open(dict_file, 'rb'))
    config.code_size = len(vocab)
    num_tree_nodes = get_rootCode(os.path.join(tree_dir, 'level2.pk')) + 1
    config.num_tree_nodes = num_tree_nodes
    class_vocab = pickle.load(open(class_dict_file, 'rb'))
    config.num_ccs_classes = len(class_vocab)
    visit_class_vocab = pickle.load(open(visit_class_dict_file, 'rb'))
    config.num_visit_classes = len(visit_class_vocab)

    config.add_dag = args.add_dag
    config.lamda = args.lamda

    if args.init_nodes:
        desc_tokenize = DescTokenizer(code2desc_file)
        tokens = desc_tokenize.tokenize(desc_tokenize.code2desc.keys())[1]
        tokens = torch.tensor(tokens).unsqueeze(0).long().to(device)
        config.tokens = tokens
    config.init_nodes = args.init_nodes

    max_seqs_len = 0
    for seq in inputs:
        if len(seq) > max_seqs_len:
            max_seqs_len = len(seq)
    config.max_position_embeddings = max_seqs_len

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data
    data_dict = dict()
    train_set, valid_set, test_set = load_data(inputs, labels, labels_visit, args.train_ratio)

    buf = '{} Total examples = {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), len(inputs))
    print2file(buf, log_file)

    train_dataset = FTDataset(train_set[0], train_set[1], train_set[2])
    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                   collate_fn=lambda batch: collate_pt(batch, config.num_ccs_classes,
                                                                       config.num_visit_classes),
                                   num_workers=0, shuffle=True)
    size_train_data = len(train_set[0])
    num_train_steps = int(size_train_data / args.train_batch_size * args.num_train_epochs)

    val_dataset = FTDataset(valid_set[0], valid_set[1], valid_set[2])
    val_data_loader = DataLoader(val_dataset, batch_size=args.train_batch_size,
                                 collate_fn=lambda batch: collate_pt(batch, config.num_ccs_classes,
                                                                     config.num_visit_classes),
                                 num_workers=0, shuffle=True)
    size_val_data = len(valid_set[0])
    num_val_steps = int(size_val_data / args.train_batch_size * args.num_train_epochs)

    test_dataset = FTDataset(test_set[0], test_set[1], test_set[2])
    test_data_loader = DataLoader(test_dataset, batch_size=args.train_batch_size,
                                  collate_fn=lambda batch: collate_pt(batch, config.num_ccs_classes,
                                                                      config.num_visit_classes),
                                  num_workers=0, shuffle=True)
    size_test_data = len(test_set[0])
    num_test_steps = int(size_test_data / args.train_batch_size * args.num_train_epochs)

    data_dict['train'] = [train_data_loader, size_train_data, num_train_steps]
    data_dict['val'] = [val_data_loader, size_val_data, num_val_steps]
    data_dict['test'] = [test_data_loader, size_test_data, num_test_steps]

    model = NextDxPrediction(config)
    model.to(device)

    params_to_update = model.parameters()
    optimizer = optim.Adadelta(params_to_update, lr=args.learning_rate)

    fout = open(os.path.join(output_dir, "loss.{}".format(datetime.datetime.now())), 'w')
    best_accuracy_at_top_5 = 0
    epoch_duration = 0.0
    global_step = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:

            buf = '{} ********** Running {} on epoch({}/{}) ***********'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                                   phase, epoch+1, int(args.num_train_epochs))
            print2file(buf, log_file)
            buf = '{} Num examples = {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                                data_dict[phase][1])
            print2file(buf, log_file)
            buf = '{}  Batch size = {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                args.train_batch_size)
            print2file(buf, log_file)
            buf = '{}  Num steps = {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                               data_dict[phase][2])
            print2file(buf, log_file)

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            data_iter = iter(data_dict[phase][0])
            tr_loss = 0
            accuracy_ls = []
            precision_ls = []
            start_time = time.time()
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(data_iter, desc="Iteration")):
                batch = {k: t.to(device) for k, t in batch.items()}

                input_ids = batch['input']
                visit_mask = batch['visit_mask']
                code_mask = batch['code_mask']
                labels_visit = batch['labels_visit']
                labels_entity = batch['labels_entity']

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = model(input_ids,
                                     visit_mask,
                                     code_mask,
                                     labels_visit,
                                     output_attentions=True)
                        loss.backward()

                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                        optimizer.step()
                        global_step += 1

                        fout.write("{}\n".format(loss.item()))
                        tr_loss += loss.item()
                        nb_tr_examples += input_ids.size(0)
                        nb_tr_steps += 1
                    else:
                        outputs, _, _ = model(input_ids,
                                              visit_mask,
                                              code_mask,
                                              output_attentions=True)
                        predicts = outputs.cpu().detach().numpy()
                        trues = labels_visit.cpu().numpy()
                        predicts = predicts.reshape(-1, predicts.shape[-1])
                        trues = trues.reshape(-1, trues.shape[-1])

                        recalls = Evaluation.visit_level_precision_at_k(trues, predicts)
                        accuracy = Evaluation.code_level_accuracy_at_k(trues, predicts)
                        precision_ls.append(recalls)
                        accuracy_ls.append(accuracy)

            duration = time.time() - start_time
            if phase == 'train':
                fout.write("train loss {} on epoch {}\n".format(epoch, tr_loss/nb_tr_steps))
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(output_dir, "pytorch_model.bin_{}".format(epoch))
                torch.save(model_to_save, output_model_file)
                buf = '{} {} Loss: {:.4f}, Duration: {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), phase, tr_loss/nb_tr_steps, duration)
                print2file(buf, log_file)
                epoch_duration += duration
            else:
                epoch_precision = (np.array(precision_ls)).mean(axis=0)
                buf = '{} {} Precision: {}, Duration: {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), phase, epoch_precision, duration)
                print2file(buf, log_file)
                epoch_accuracy = (np.array(accuracy_ls)).mean(axis=0)
                buf = '{} {} Accuracy: {}, Duration: {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), phase, epoch_accuracy, duration)
                print2file(buf, log_file)
                if phase == 'val' and epoch_accuracy[0] > best_accuracy_at_top_5:
                    best_accuracy_at_top_5 = epoch_accuracy[0]
                    best_model_wts = copy.deepcopy(model.state_dict())
    fout.close()

    buf = '{} Training complete in {:.0f}m {:.0f}s'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                           epoch_duration // 60, epoch_duration % 60)
    print2file(buf, log_file)

    buf = '{} Best accuracy at top 5: {:4f}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                    best_accuracy_at_top_5)
    print2file(buf, log_file)

    # load best model weights
    model.load_state_dict(best_model_wts)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    buf = '{} Save the best model to {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), output_model_file)
    print2file(buf, log_file)
    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    torch.save(model_to_save, output_model_file)


if __name__ == "__main__":
    main()


