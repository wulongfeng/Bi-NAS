import numpy as np
import os
import os.path
import sys
import shutil
import torch
import torch.nn as nn
import torch.utils
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import shuffle
from collections import defaultdict
import pickle
# from models_binary import PRIMITIVES_BINARY, Network_Search
# from models_triple import PRIMITIVES_TRIPLE, Network_Search_Triple

from torch.utils.tensorboard import SummaryWriter
from model_NAR import NAR
from datetime import datetime

parent_path = os.path.dirname(os.path.realpath(__file__))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def sample_arch():
    arch = {}
    arch['mlp'] = {}
    arch['mlp']['p'] = nn.Sequential(
        nn.Linear(1, 8),
        nn.Tanh(),
        nn.Linear(8, 1)).cuda()
    arch['mlp']['q'] = nn.Sequential(
        nn.Linear(1, 8),
        nn.Tanh(),
        nn.Linear(8, 1)).cuda()
    arch['binary'] = PRIMITIVES_BINARY[np.random.randint(len(PRIMITIVES_BINARY))]
    return arch


def sample_arch_triple():
    arch = {}
    arch['mlp'] = {}
    arch['mlp']['p'] = nn.Sequential(
        nn.Linear(1, 8),
        nn.Tanh(),
        nn.Linear(8, 1)).cuda()
    arch['mlp']['q'] = nn.Sequential(
        nn.Linear(1, 8),
        nn.Tanh(),
        nn.Linear(8, 1)).cuda()
    arch['mlp']['r'] = nn.Sequential(
        nn.Linear(1, 8),
        nn.Tanh(),
        nn.Linear(8, 1)).cuda()
    arch['triple'] = PRIMITIVES_TRIPLE[np.random.randint(len(PRIMITIVES_TRIPLE))]
    return arch


def update_arch(arch, cfg):
    flag = 0
    for p in arch.parameters():
        num = p.view(-1).size(0)
        p.data.add_(-p).add_(torch.tensor(cfg[flag:flag + num]).float().cuda().view(p.size()))
        flag += num


def load_arch(num_users, num_items, args):
    arch = {}
    arch['mlp'] = {}
    with open(os.path.join('experiments', args.dataset, args.arch, 'log.txt'), 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.split()
            if 'genotype:' in line:
                arch['binary'] = line[-1]

    model = Network_Search(num_users, num_items, args.embedding_dim, args.weight_decay)
    model.load_state_dict(torch.load(os.path.join('experiments', args.dataset, args.arch, 'model.pt')))
    arch['mlp']['p'] = model.mlp_p
    arch['mlp']['q'] = model.mlp_q
    return arch


def load_arch_triple(num_ps, num_qs, num_rs, args):
    arch = {}
    arch['mlp'] = {}
    with open(os.path.join('experiments', args.dataset, args.arch, 'log.txt'), 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.split()
            if 'genotype:' in line:
                arch['triple'] = line[-1]

    model = Network_Search_Triple(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay)
    model.load_state_dict(torch.load(os.path.join('experiments', args.dataset, args.arch, 'model.pt')))
    arch['mlp']['p'] = model.mlp_p
    arch['mlp']['q'] = model.mlp_q
    arch['mlp']['r'] = model.mlp_r
    return arch


def get_data_queue(args):
    users, items, labels = [], [], []
    if args.dataset == 'ml-100k':
        data_path = os.path.join(args.data, 'ml-100k', 'u.data')
    elif args.dataset == 'ml-1m':
        data_path = os.path.join(args.data, 'ml-1m', 'ratings.dat')
    elif args.dataset == 'ml-10m':
        data_path = os.path.join(args.data, 'ml-10m', 'ratings.dat')
    elif args.dataset == 'youtube-small':
        data_path = os.path.join(args.data, 'youtube-weighted-small.npy')

    if 'ml' in args.dataset:
        # movielens dataset
        with open(data_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if args.dataset == 'ml-100k':
                    line = line.split()
                elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
                    line = line.split('::')
                users.append(int(line[0]) - 1)
                items.append(int(line[1]) - 1)
                labels.append(float(line[2]))
        labels = StandardScaler().fit_transform(np.reshape(labels, [-1, 1])).flatten().tolist()

        print('user', max(users), min(users))
        print('item', max(items), min(items))

        users, items, labels = shuffle(users, items, labels)
        indices = list(range(len(users)))
        num_train = int(len(users) * args.train_portion)
        num_valid = int(len(users) * args.valid_portion)

        if not args.mode == 'libfm':
            data_queue = torch.utils.data.TensorDataset(torch.tensor(users),
                                                        torch.tensor(items), torch.tensor(labels))

            train_queue = torch.utils.data.DataLoader(data_queue, batch_size=args.batch_size,
                                                      sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                          indices[:num_train]), pin_memory=True)

            valid_queue = torch.utils.data.DataLoader(data_queue, batch_size=args.batch_size,
                                                      sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                          indices[num_train:num_train + num_valid]), pin_memory=True)

            test_queue = torch.utils.data.DataLoader(data_queue, batch_size=args.batch_size,
                                                     sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                         indices[num_train + num_valid:]), pin_memory=True)

        else:
            # prepare data format for libfm
            data_queue = []
            for i in range(len(users)):
                data_queue.append({'user': str(users[i]), 'item': str(items[i])})

            v = DictVectorizer()
            data_queue = v.fit_transform(data_queue)
            train_queue = [data_queue[:num_train], np.array(labels[:num_train])]
            valid_queue = [data_queue[num_train:num_train + num_valid],
                           np.array(labels[num_train:num_train + num_valid])]
            test_queue = [data_queue[num_train + num_valid:], np.array(labels[num_train + num_valid:])]

    else:
        # 3-d dataset
        [ps, qs, rs, labels] = np.load(data_path).tolist()
        labels = StandardScaler().fit_transform(np.reshape(labels, [-1, 1])).flatten().tolist()

        ps = [int(i) for i in ps]
        qs = [int(i) for i in qs]
        rs = [int(i) for i in rs]
        print('p', max(ps), min(ps))
        print('q', max(qs), min(qs))
        print('r', max(rs), min(rs))

        ps, qs, rs, labels = shuffle(ps, qs, rs, labels)
        indices = list(range(len(ps)))
        num_train = int(len(ps) * args.train_portion)
        num_valid = int(len(ps) * args.valid_portion)

        if not args.mode == 'libfm':
            data_queue = torch.utils.data.TensorDataset(torch.tensor(ps), torch.tensor(qs),
                                                        torch.tensor(rs), torch.tensor(labels))

            train_queue = torch.utils.data.DataLoader(data_queue, batch_size=args.batch_size,
                                                      sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                          indices[:num_train]), pin_memory=True)

            valid_queue = torch.utils.data.DataLoader(data_queue, batch_size=args.batch_size,
                                                      sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                          indices[num_train:num_train + num_valid]), pin_memory=True)

            test_queue = torch.utils.data.DataLoader(data_queue, batch_size=args.batch_size,
                                                     sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                         indices[num_train + num_valid:]), pin_memory=True)

        else:
            # prepare data format for libfm
            data_queue = []
            for i in range(len(ps)):
                data_queue.append({'p': str(ps[i]), 'q': str(qs[i]), 'r': str(rs[i])})

            v = DictVectorizer()
            data_queue = v.fit_transform(data_queue)
            train_queue = [data_queue[:num_train], np.array(labels[:num_train])]
            valid_queue = [data_queue[num_train:num_train + num_valid],
                           np.array(labels[num_train:num_train + num_valid])]
            test_queue = [data_queue[num_train + num_valid:], np.array(labels[num_train + num_valid:])]

    return train_queue, valid_queue, test_queue


def shuffle(train_users, train_users_feature, train_pos_items, train_pos_items_feature, train_neg_items,
            train_neg_items_feature):
    train_records_num = len(train_users)
    index = np.array(range(train_records_num)).astype(int)
    np.random.shuffle(index)
    input_user = list(np.array(train_users)[index])
    input_user_feature = list(np.array(train_users_feature)[index])
    input_pos_item = list(np.array(train_pos_items)[index])
    input_pos_item_feature = list(np.array(train_pos_items_feature)[index])
    input_neg_item = list(np.array(train_neg_items)[index])
    input_neg_item_feature = list(np.array(train_neg_items_feature)[index])

    return input_user, input_user_feature, input_pos_item, input_pos_item_feature, input_neg_item, input_neg_item_feature


class Evaluate(object):
    def __init__(self, topk):
        self.Top_K = topk

    def MAP(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            tmp = 0
            hit = 0
            for j in range(len(fit)):
                if fit[j] in v:
                    hit += 1
                    tmp += hit / (j + 1)
            result.append(tmp)
        # print("result of map:{}".format(result))
        return np.array(result).mean()

    def MRR(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            tmp = 0
            for j in range(len(fit)):
                if fit[j] in v:
                    tmp = 1 / (j + 1)
                    break
            result.append(tmp)
        return np.array(result).mean()

    def NDCG(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            temp = 0
            Z_u = 0

            for j in range(min(len(fit), len(v))):
                Z_u = Z_u + 1 / np.log2(j + 2)
            for j in range(len(fit)):
                if fit[j] in v:
                    temp = temp + 1 / np.log2(j + 2)

            if Z_u == 0:
                temp = 0
            else:
                temp = temp / Z_u
            result.append(temp)
        return np.array(result).mean()

    def top_k(self, ground_truth, pred):
        p_total = []
        r_total = []
        f_total = []
        hit_total = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            cross = float(len([i for i in fit if i in v]))
            p = cross / len(fit)
            r = cross / len(v)
            if cross > 0:
                f = 2.0 * p * r / (p + r)
            else:
                f = 0.0
            hit = 1.0 if cross > 0 else 0.0
            p_total.append(p)
            r_total.append(r)
            f_total.append(f)
            hit_total.append(hit)
        return np.array(p_total).mean(), np.array(r_total).mean(), np.array(f_total).mean(), np.array(hit_total).mean()

    def evaluate(self, ground_truth, pred):
        # pred { uid: {iid : score, }}
        # ground_truth { uid: [tid...] }
        # map, mrr, p, r, f1, hit, ndcg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        sorted_pred = {}
        for k, v in pred.items():
            sorted_pred[k] = sorted(v.items(), key=lambda item: item[1])[::-1]

        # Protocol : pred { uid: [(tid, score)...)] }
        # Protocol : ground_truth { uid: [tid...] }
        p, r, f1, hit = self.top_k(ground_truth, sorted_pred)
        map = self.MAP(ground_truth, sorted_pred)
        mrr = self.MRR(ground_truth, sorted_pred)
        ndcg = self.NDCG(ground_truth, sorted_pred)
        return map, mrr, p, r, f1, hit, ndcg


def load_word_embedding(debug=False):
    if debug:
        return {}, np.zeros((10000, 300))
    else:
        lines = open(os.path.join(parent_path, 'data/glove.6B.300d.txt')).readlines()
        data = []
        word_dict = {}
        for idx, line in enumerate(lines):
            tokens = line.strip('\n')
            # word, vec = tokens.split(' ')
            tokens = tokens.split(' ')
            word = tokens[0]
            vec_nums = tokens[1:]
            # vec_nums = vec.split(' ')
            word_dict[word] = idx
            temp_ = [float(i) for i in vec_nums]
            assert len(temp_) == 300
            data.append(temp_)
        data = np.array(data)
        assert data.shape[1] == 300
        print("Loaded data. #shape = " + str(data.shape))
        print(" #words = %d " % (len(word_dict)))
        return word_dict, data


def init_model(args, data):
    evaluator = Evaluate(args.K)
    model, i_model = None, None
    args.out_loop = 1
    model = NAR(args, data.user_num, data.item_num, args.dim, data.w2v_feat_np, data.user_feat_dim)

    # print("Number of Params:{}".format(model.num_params))
    model.args = args
    return evaluator, model, i_model


def init_logger(args):
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")

    writer = None
    title = '{}_{}_{}_{}_O:{}_I:{}_drop_{}_nnOut_{}'.format(
        args.dataset_str, args.model_name, args.lr, args.K, args.out_loop,
        args.epochs, args.dropout, args.use_output_layer)
    if 'CAR' in args.model_name:
        title += '_reg_adv_{}_adv_epoch_{}_eps_{}_reg_{}'.format(
            args.reg_adv, args.adv_epoch, args.eps, args.reg)

    if args.tb_log:
        writer = SummaryWriter(comment=title)
    if args.txt_log:
        writer = '/log/{}_{}.txt'.format(title, dt_string)
    return writer


def save_model_func(args, model):
    save_folder = parent_path + '/saved_model'
    file_name = 'best_{}_{}.pkl'.format(args.model_name, args.dataset_str)
    save_path = '{}/{}'.format(save_folder, file_name)
    torch.save(model.state_dict(), save_path)


def vis_func(args, data, model):
    word_list = []
    # TODO: the word has been filtered, and the order may be different
    for word, id in data.feature_id_dict.items():
        word_list.append(word)

    user_word_score_dict, item_word_score_dict = model.vis(data)

    sorted_user_pertub_dict = defaultdict(list)
    word_sorted_dict = defaultdict(list)
    num_words = len(word_list)
    for user, score_v in user_word_score_dict.items():
        temp_dict = {}
        for word, score in zip(word_list, list(score_v)):
            temp_dict[word] = score

        word_sorted_dict[user] = sorted(temp_dict.items(), key=lambda x: x[0], reverse=False)

        if args.model_name == 'NAR':
            temp_list = sorted(temp_dict.items(), key=lambda x: x[1], reverse=False)
            sorted_user_pertub_dict[user] = temp_list
        elif args.model_name == 'CNR':
            temp_list = sorted(temp_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_user_pertub_dict[user] = temp_list
        elif args.model_name == 'CAR':
            temp_list = sorted(temp_dict.items(), key=lambda x: abs(x[1]), reverse=False)
            sorted_user_pertub_dict[user] = temp_list

    vis_matrix = np.zeros((len(word_sorted_dict), num_words))
    for u_idx in range(len(word_sorted_dict)):
        vis_matrix[u_idx] = np.array([score for (w, score) in word_sorted_dict[u_idx]])  # word_sorted_dict[u_idx]

    with open(parent_path + '/saved_vis_matrix/{}_{}_user_vis_matrix.pkl'.format(args.dataset_str, args.model_name), 'wb') as file:
        pickle.dump(vis_matrix, file)

    item_word_sorted_dict = defaultdict(list)
    # i_mark = 0
    print("the number of items with word score:{}".format(len(item_word_score_dict)))
    for item, score_v in item_word_score_dict.items():
        # i_mark += 1
        temp_dict = {}
        for word, score in zip(word_list, list(score_v)):
            temp_dict[word] = score
        item_word_sorted_dict[item] = sorted(temp_dict.items(), key=lambda x: x[0], reverse=False)
    # if i_mark < 10:
    #    print("item:{} and value:{}".format(item, item_word_sorted_dict[item]))

    item_vis_matrix = np.zeros((len(item_word_sorted_dict), num_words))
    print("shape of item_vis_matrix :{}".format(item_vis_matrix.shape))
    for i_idx in range(len(item_word_sorted_dict)):
        # print("idx :{}, scores:{}".format(i_idx, [score for (w, score) in item_word_sorted_dict[i_idx]]))
        if i_idx in item_word_sorted_dict:
            item_vis_matrix[i_idx] = np.array(
                [score for (w, score) in item_word_sorted_dict[i_idx]])  # word_sorted_dict[u_idx]

    with open(parent_path + '/saved_vis_matrix/{}_{}_item_vis_matrix.pkl'.format(args.dataset_str, args.model_name),
              'wb') as file:
        pickle.dump(item_vis_matrix, file)

    user_feature_sentiment = data.user_feature_sentiment
    user_feature_sentiment_groundtruth = defaultdict()
    for uid, fea_sent in user_feature_sentiment.items():
        user_feature_sentiment_groundtruth[uid] = list(user_feature_sentiment[uid].keys())

    user_feature_perturb = defaultdict()
    for uid, fea_purb in sorted_user_pertub_dict.items():
        user_feature_perturb[uid] = [data.feature_id_dict[word] for word, _ in fea_purb]

    p, r, f1, ndcg = explanation_evaluate(user_feature_sentiment_groundtruth, user_feature_perturb, k=10)
    print('Evaluate Explanation (User sentiment oriented) --> Dataset: {} Model: {} precision: {:#.4g}, recall: '
          '{:#.4g}, f1: {:#.4g}, ndcg: {:#.4g}'.format(args.dataset_str, args.model_name, p, r, f1, ndcg))


def explanation_evaluate(gt, pred, k=10):
    p_total, r_total, f1_total = list(), list(), list()
    ndcg_total = list()
    for uid, wids in gt.items():
        fit = pred[uid][:k]

        # precision, recall, f1
        cross = float(len([i for i in fit if i in wids]))
        p = cross / len(fit)
        r = cross / len(wids)
        if cross > 0:
            f1 = 2.0 * p * r / (p + r)
        else:
            f1 = 0.0

        p_total.append(p)
        r_total.append(r)
        f1_total.append(f1)

        # ndcg
        temp = 0
        Z_u = 0
        for j in range(min(len(fit), len(wids))):
            Z_u = Z_u + 1 / np.log2(j + 2)
        for j in range(len(fit)):
            if fit[j] in set(wids):
                temp = temp + 1 / np.log2(j + 2)

        if Z_u == 0:
            temp = 0
        else:
            temp = temp / Z_u
        ndcg_total.append(temp)

    return np.array(p_total).mean(), np.array(r_total).mean(), np.array(f1_total).mean(), np.array(ndcg_total).mean()
