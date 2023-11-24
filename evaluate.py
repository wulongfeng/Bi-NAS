import torch
import torch.nn.functional as F
import numpy as np
import math
import random
from tqdm import tqdm
import time

def evaluate_nar(model, data):
	pred = dict()
	ground_truth = dict()
	model.eval()
	test_user_list = list(data.ground_truth_user_items_feature_dict.keys())
	sampled_test_user_list = random.sample(test_user_list, min(1000, len(test_user_list)))
	print("testing...")
	xs, ys = [], []
	with torch.no_grad():
		#for u in tqdm(sampled_test_user_list):
		for u in sampled_test_user_list:
			items = data.ground_truth_user_items_dict[u]
			#print("u:{}, items:{}".format(u, items)
			label = []
			for i in items:
				label.append(data.user_item_label[u][i])

			features = data.ground_truth_user_items_feature_dict[u]
			u_extend = [u] * len(items)
			u_feature_extend = data.user_feature_all[u].repeat(len(items),1) # [data.user_feature_all[u].cpu().numpy()] * len(items)

			user_id_t = torch.LongTensor(u_extend).cuda()
			user_feature_t = u_feature_extend #torch.FloatTensor(u_feature_extend).to(model.args.device)
			item_id_t = torch.LongTensor(items).cuda()
			item_feature_t = torch.cat(features, dim=0) # torch.FloatTensor(features).to(model.args.device)

			#scores = model.forward(user_id_t, user_feature_t, item_id_t, item_feature_t).cpu()
			#pred[u] = dict(zip(items, scores))
			#ground_truth[u] = data.ground_truth_user_items_dict[u]
			inferences, _ = model(user_id_t.cuda(), user_feature_t.cuda(), item_id_t.cuda(), item_feature_t.cuda())
			#labels = torch.FloatTensor([random.uniform(-2.1, 1.9) for _ in range(len(items))]).cuda()
			labels = torch.FloatTensor(label).cuda()
			xs.append(inferences.flatten())
			ys.append(labels.cuda())
		mse = F.mse_loss(torch.cat(xs), torch.cat(ys))
		rmse = torch.sqrt(mse)
	return rmse.cpu().detach().item()


def evaluate_nar_ce(model, data):
	pred = dict()
	ground_truth = dict()
	model.eval()
	test_user_list = list(data.ground_truth_user_items_feature_dict.keys())
	sampled_test_user_list = random.sample(test_user_list, min(1000, len(test_user_list)))
	print("testing...")
	xs, ys = [], []
	with torch.no_grad():
		#for u in tqdm(sampled_test_user_list):
		for u in sampled_test_user_list:
			items = data.ground_truth_user_items_dict[u]
			#print("u:{}, items:{}".format(u, items)
			label = []
			for i in items:
				label.append(data.user_item_label[u][i])

			features = data.ground_truth_user_items_feature_dict[u]
			u_extend = [u] * len(items)
			u_feature_extend = data.user_feature_all[u].repeat(len(items),1) # [data.user_feature_all[u].cpu().numpy()] * len(items)

			user_id_t = torch.LongTensor(u_extend).cuda()
			user_feature_t = u_feature_extend #torch.FloatTensor(u_feature_extend).to(model.args.device)
			item_id_t = torch.LongTensor(items).cuda()
			item_feature_t = torch.cat(features, dim=0) # torch.FloatTensor(features).to(model.args.device)

			#scores = model.forward(user_id_t, user_feature_t, item_id_t, item_feature_t).cpu()
			#pred[u] = dict(zip(items, scores))
			#ground_truth[u] = data.ground_truth_user_items_dict[u]
			inferences, _ = model(user_id_t.cuda(), user_feature_t.cuda(), item_id_t.cuda(), item_feature_t.cuda())
			#labels = torch.FloatTensor([random.uniform(-2.1, 1.9) for _ in range(len(items))]).cuda()
			labels = torch.FloatTensor(label).cuda()
			xs.append(inferences.flatten())
			ys.append(labels.cuda())
		mse = F.mse_loss(torch.cat(xs), torch.cat(ys))
		rmse = torch.sqrt(mse)
	return rmse.cpu().detach().item()


def test_training_ce(data, model, evaluator):
	pred = dict()
	ground_truth = dict()
	train_user_list = list(data.train_user_positive_items_dict.keys())
	sampled_train_user_list = random.sample(train_user_list, min(5000, len(train_user_list)))
	with torch.no_grad():
		for u in sampled_train_user_list:
			items = data.train_user_items_dict[u]
			features = data.train_user_items_feature_dict[u]
			u_extend = [u] * len(items)
			u_feature_extend = data.user_feature_all[u].repeat(len(items),1) # [data.user_feature_all[u].cpu().numpy()] * len(items)

			user_id_t = torch.LongTensor(u_extend).cuda()
			user_feature_t = u_feature_extend #torch.FloatTensor(u_feature_extend).to(model.args.device)
			item_id_t = torch.LongTensor(items).cuda()
			item_feature_t = torch.cat(features, dim=0) # torch.FloatTensor(features).to(model.args.device)

			scores = model(user_id_t, user_feature_t, item_id_t, item_feature_t).cpu()
			pred[u] = dict(zip(items, scores))
			ground_truth[u] = data.train_user_positive_items_dict[u]
			#print("training u:{}".format(u))
			#print("training ground_truth:{}".format(ground_truth[u]))
			#print("training pred:{}".format(pred[u]))
	map, mrr, p, r, f1, hit, ndcg = evaluator.evaluate(ground_truth, pred)
	return map, mrr, p, r, f1, hit, ndcg


def test_nar_ce(data, model, evaluator):
	pred = dict()
	ground_truth = dict()
	losses = []
	model.eval()
	test_user_list = list(data.compute_user_items_dict.keys())
	sampled_test_user_list = random.sample(test_user_list, min(5000, len(test_user_list)))
	print("testing...")
	test_t = time.time()
	with torch.no_grad():
		for u in tqdm(sampled_test_user_list):
			items = data.compute_user_items_dict[u]
			features = data.compute_user_items_feature_dict[u]
			u_extend = [u] * len(items)
			u_feature_extend = data.user_feature_all[u].repeat(len(items),1) # [data.user_feature_all[u].cpu().numpy()] * len(items)

			user_id_t = torch.LongTensor(u_extend).cuda()
			user_feature_t = u_feature_extend #torch.FloatTensor(u_feature_extend).to(model.args.device)
			item_id_t = torch.LongTensor(items).cuda()
			item_feature_t = torch.cat(features, dim=0) # torch.FloatTensor(features).to(model.args.device)

			scores = model.forward(user_id_t, user_feature_t, item_id_t, item_feature_t).cpu()
			pred[u] = dict(zip(items, scores))
			ground_truth[u] = data.ground_truth_user_items_dict[u]

			# for loss
			ground_len = len(ground_truth[u])
			ground_u = [u] * ground_len

			pos_user = torch.LongTensor(ground_u).cuda()
			pos_user_feature = data.user_feature_all[u].repeat(ground_len,1)

			pos_items_train = torch.LongTensor(ground_truth[u]).cuda()
			pos_item_feat = []
			for item in ground_truth[u]:
				pos_item_feat.append(data.item_feature_all[item].reshape(1,-1))
			pos_items_feat_train = torch.cat(pos_item_feat, dim=0)

			candidates = list(set(items) - set(ground_truth[u]))
			ground_neg_item = random.sample(candidates, ground_len)
			neg_items_train = torch.LongTensor(ground_neg_item).cuda()
			neg_item_feat = []
			for item in ground_neg_item:
				neg_item_feat.append(data.item_feature_all[item].reshape(1,-1))
			neg_items_feat_train = torch.cat(neg_item_feat, dim=0)
			loss = model.compute_loss(pos_user, pos_user_feature, pos_items_train, pos_items_feat_train, neg_items_train, neg_items_feat_train)
			losses.append(loss.item())
			#print("test pred:{}".format(pred[u]))
			#print("test ground_truth:{}".format(ground_truth[u]))
	map, mrr, p, r, f1, hit, ndcg = evaluator.evaluate(ground_truth, pred)
	print('Test used time:{}'.format(time.time() - test_t))
	return np.array(losses).mean(), map, mrr, p, r, f1, hit, ndcg


def evaluate(model, test_queue):
    model.eval()
    xs, ys = [], []
    with torch.no_grad():
        for users, items, labels in test_queue:
            inferences, _ = model(users.cuda(), items.cuda())
            xs.append(inferences.flatten())
            ys.append(labels.cuda())
        mse = F.mse_loss(torch.cat(xs), torch.cat(ys))
        rmse = torch.sqrt(mse)
    return rmse.cpu().detach().item()


def evaluate_triple(model, test_queue):
	model.eval()
	xs, ys = [], []
	with torch.no_grad():
		for ps, qs, rs, labels in test_queue:
			inferences, _ = model(ps.cuda(), qs.cuda(), rs.cuda())
			xs.append(inferences.flatten())
			ys.append(labels.cuda())
		mse = F.mse_loss(torch.cat(xs), torch.cat(ys))
		rmse = torch.sqrt(mse)
	return rmse.cpu().detach().item()


def evaluate_hr_ndcg(model, test_queue, topk=10):
	model.eval()
	with torch.no_grad():
		users, items, _ = test_queue
		users = users.cpu().tolist()
		hrs, ndcgs = [], []
		
		inferences_dict = {}
		
		users_all, items_all = [], []
		for user in list(set(users)):
			users_all += [user] * model.num_items
			items_all += list(range(model.num_items))
		inferences, _ = model(torch.tensor(users_all).cuda(), torch.tensor(items_all).cuda())
		inferences = inferences.detach().cpu().tolist()
		for i, user in enumerate(list(set(users))):
			inferences_dict[user] = inferences[i*model.num_items:(i+1)*model.num_items]

		for i, user in enumerate(users):
			inferences = inferences_dict[user]
			score = inferences[items[i]]
			rank = 0
			for s in inferences:
				if score < s:
					rank += 1
			if rank < topk:
				hr = 1.0
				ndcg = math.log(2) / math.log(rank+2)
			else:
				hr = 0.0
				ndcg = 0.0
			hrs.append(hr)
			ndcgs.append(ndcg)
	return np.mean(hrs), np.mean(ndcgs)
