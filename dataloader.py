import os
import time
import numpy as np
import pickle
import random
import copy

import torch
import torch.utils.data as tdata
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from utils import load_word_embedding

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

folder_dict = {}
folder_dict['video'] = 'Amazon_Instant_Video/'
folder_dict['Beauty'] = 'Beauty/'
folder_dict['Clothing'] = 'Clothing_Shoes_and_Jewelry/'
# folder_dict['Music'] = 'Digital_Music/'
folder_dict['Musical'] = 'Musical_Instruments/'


class DataLoader():
    def __init__(self, args):
        self.args = args
        self.path = args.data_path + folder_dict[args.dataset_str]
        print("data_path:{}".format(args.data_path))
        print(self.path)
        self.statistics = dict()
        self.user_id_dict = pickle.load(open(self.path + "user_id_dict", "rb"))
        self.item_id_dict = pickle.load(open(self.path + "item_id_dict", "rb"))
        self.feature_id_dict = pickle.load(open(self.path + "feature_id_dict", "rb"))

        self.statistics['user_number'] = len(self.user_id_dict.keys())
        self.statistics['item_number'] = len(self.item_id_dict.keys())
        self.statistics['feature_number'] = len(self.feature_id_dict.keys())  # feature_id_dict： word to id
        self.user_num = self.statistics['user_number']
        self.item_num = self.statistics['item_number']

        self.user_feature_attention = pickle.load(open(self.path + "u_fea_{}".format(args.user_feat_type), "rb"))
        self.item_feature_quality = pickle.load(open(self.path + "i_fea_{}".format(args.user_feat_type), "rb"))
        self.train_user_positive_items_dict = pickle.load(open(self.path + "train_user_positive_items_dict", "rb"))
        # self.train_user_negative_items_dict = pickle.load(open(self.path + "train_user_negative_items_dict", "rb"))
        self.ground_truth_user_items_dict = pickle.load(open(self.path + "test_ground_truth_user_items_dict", "rb"))
        self.compute_user_items_dict = pickle.load(open(self.path + "test_compute_user_items_dict", "rb"))
        # use as gt for vis eval
        self.user_feature_sentiment = pickle.load(open(self.path + "user_feature_sentiment", "rb"))

        self.w2v_feat_np = None
        #self.load_rating()
        self.init_data()
        self.generate_train_corpus()
        self.generate_validation_corpus()
        print("Feature Dim:{}".format(self.item_feat_dim))

    def load_rating(self):
        users, items, labels = [], [], []
        self.user_item_label = defaultdict(lambda: defaultdict(float))
        # user_id_dict = pickle.load(open(self.path + "user_id_dict", "rb"))
        # item_id_dict = pickle.load(open(self.path + "item_id_dict", "rb"))
        # print("number of users: {}".format(len(user_id_dict)))
        # print("number of items: {}".format(len(item_id_dict)))

        reviews = pickle.load(open(self.path + "reviews.pickle", "rb"))
        print("records of reviews: {}".format(len(reviews)))

        for res_tuple in reviews:
            user, item, rating = res_tuple['user'], res_tuple['item'], res_tuple['rating']
            if user in self.user_id_dict and item in self.item_id_dict:
                users.append(self.user_id_dict[user])
                items.append(self.item_id_dict[item])
                labels.append(float(rating))

        assert len(users) == len(items)
        assert len(users) == len(labels)
        labels = StandardScaler().fit_transform(np.reshape(labels, [-1,1])).flatten().tolist()

        for i in range(len(users)):
            self.user_item_label[users[i]][items[i]] = labels[i]
        print("length of the user_item_label: {}".format(len(self.user_item_label)))


    def init_data(self):
        self.user_all = []
        self.item_all = []
        self.user_feature_all = []
        self.item_feature_all = []
        self.pos_item_all = []
        self.train_user_all = []
        #self.train_label_all = []

        # read data to tensor
        for i in range(len(self.user_feature_attention)):
            self.user_all.append(i)
            assert i in self.user_feature_attention
            self.user_feature_all.append(self.user_feature_attention[i])

        for i in range(len(self.item_feature_quality)):
            assert i in self.item_feature_quality
            self.item_feature_all.append(self.item_feature_quality[i])

        self.user_feature_all = torch.FloatTensor(self.user_feature_all).to(self.args.device)
        self.item_feature_all = torch.FloatTensor(self.item_feature_all).to(self.args.device)

        # pruning
        assert self.user_feature_all.shape[1] == self.item_feature_all.shape[1]
        DIM=self.args.feat_selection_num
        if self.user_feature_all.shape[1] > DIM:
            user_var_dims = torch.var(self.user_feature_all, dim=0)
            _, user_feat_filter_idx = torch.topk(user_var_dims, DIM)
            self.user_feature_all = self.user_feature_all[:, user_feat_filter_idx]
            self.user_feat_filter_idx = user_feat_filter_idx
            item_var_dims = torch.var(self.item_feature_all, dim=0)
            _, item_feat_filter_idx = torch.topk(item_var_dims, DIM)
            self.item_feature_all = self.item_feature_all[:, item_feat_filter_idx]
            self.item_feat_filter_idx = item_feat_filter_idx
            item_var_dims_set = set(list(item_feat_filter_idx.cpu().numpy()))

            # create filtered word dict
            filtered_feature_id_dict = {}
            for idx, (k,v) in enumerate(self.feature_id_dict.items()):
                if idx in item_var_dims_set:
                    filtered_feature_id_dict[k] = v
            self.feature_id_dict = filtered_feature_id_dict

        else:
            item_var_dims_set = set(range(self.user_feature_all.shape[1]))
            self.user_feat_filter_idx = torch.tensor(range(self.user_feature_all.shape[1])).to(self.args.device)
            self.item_feat_filter_idx = torch.tensor(range(self.item_feature_all.shape[1])).to(self.args.device)

        self.item_feat_dim = self.item_feature_all.shape[1]
        self.user_feat_dim = self.user_feature_all.shape[1]

        if self.args.model_name == 'NAR':
            print("load word to vec data.")
            w2v_feat = []
            t = time.time()
            w2v_dict, w2v_matrix = load_word_embedding(debug=False)
            for word, id in tqdm(self.feature_id_dict.items()):
                if id not in item_var_dims_set:
                    print("id:{} not in dict".format(id))
                    exit(1)
                    continue
                if word in w2v_dict:
                    idx = w2v_dict[word]
                    w2v_feat.append(w2v_matrix[idx])
                elif ' ' in word:
                    tokens = word.split(' ')
                    temp = np.zeros(300, )
                    for token in w2v_dict:
                        if tokens in w2v_matrix:
                            temp += w2v_matrix[w2v_dict[token]]
                    w2v_feat.append(temp)
                else:
                    print('{} not in dict!'.format(word))
                    w2v_feat.append(np.random.random(300, ))
            self.w2v_feat_np = np.stack(w2v_feat, axis=0)
            print("Process w2v time:{}".format(time.time() - t))

        # create loader
        print("create train loader.")
        for user, positive_items in self.train_user_positive_items_dict.items():
            assert len(positive_items) > 0
            for item in positive_items:
                pos_item_id = int(item)
                self.pos_item_all.append(pos_item_id)
                self.train_user_all.append(user)
        #        self.train_label_all.append(self.user_item_label[user][pos_item_id])
        self.pos_item_all = torch.LongTensor(self.pos_item_all).to(self.args.device)
        self.train_user_all = torch.LongTensor(self.train_user_all).to(self.args.device)
        #self.train_label_all = torch.FloatTensor(self.train_label_all).to(self.args.device)

        #self.train_dataloader = tdata.DataLoader(tdata.TensorDataset(self.train_user_all, self.pos_item_all, self.train_label_all),
        #                                        batch_size=self.args.batch_size, shuffle=True)
        self.train_dataloader = tdata.DataLoader(tdata.TensorDataset(self.train_user_all, self.pos_item_all),
                                                 batch_size=self.args.batch_size, shuffle=True)

        self.pos_item_val = []
        self.user_val = []
        for user, positive_items in self.ground_truth_user_items_dict.items():
            assert len(positive_items) > 0
            for item in positive_items:
                pos_item_id = int(item)
                self.pos_item_val.append(pos_item_id)
                self.user_val.append(user)
        self.pos_item_val = torch.LongTensor(self.pos_item_val).to(self.args.device)
        self.user_val = torch.LongTensor(self.user_val).to(self.args.device)
        self.val_dataloader = tdata.DataLoader(tdata.TensorDataset(self.user_val, self.pos_item_val),
                                                 batch_size=self.args.batch_size, shuffle=True)

    # def generate_validation_corpus(self):
    #     self.ground_truth_user_items_feature_dict = dict()
    #
    #     for user, item_list in self.ground_truth_user_items_dict.items():
    #         tmp = []
    #         for item in item_list:
    #             tmp.append(self.item_feature_all[item].reshape(1,-1))
    #         self.ground_truth_user_items_feature_dict[user] = tmp
    def generate_train_corpus(self):
        self.train_user_items_feature_dict = dict()
        self.train_user_items_dict = dict()

        for user, item_list in self.train_user_positive_items_dict.items():
            #print("length of item list:{}, and items:{}".format(len(item_list), item_list))
            train_user_random_items = copy.deepcopy(item_list)
            if len(item_list) < 100:
                candidate_iid = list(set(range(self.item_num)) - set(item_list))
                train_user_random_items.extend(random.sample(candidate_iid, 100 - len(item_list)))
            self.train_user_items_dict[user] = train_user_random_items
            #print("length of item list:{}".format(len(item_list)))
            #print("item list:{}".format(item_list))
            tmp = []
            for item in train_user_random_items:
                tmp.append(self.item_feature_all[item].reshape(1,-1))
            self.train_user_items_feature_dict[user] = tmp

    def generate_validation_corpus(self):
        self.compute_user_items_feature_dict = dict()
        for user, item_list in self.compute_user_items_dict.items():
            tmp = []
            for item in item_list:
                tmp.append(self.item_feature_all[item].reshape(1,-1))
            self.compute_user_items_feature_dict[user] = tmp