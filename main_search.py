import os
import sys
import glob
import numpy as np
import torch
import logging
import argparse
import torch.backends.cudnn as cudnn
import torch.utils

import time
import utils
from scipy.stats import entropy

from train import train_nar_ce
from models_binary import Network_NAR
from models_triple import Network_NAR_Triple
from models_quadruple import Network_NAR_Quadruple

from evaluate import test_nar_ce, test_training_ce

from dataloader import DataLoader
from utils import Evaluate, vis_func, save_model_func
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
#print("dir path:{}".format(dir_path))
#print("parent path:{}".format(parent_path))
parser = argparse.ArgumentParser(description="Search.")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument("--data_path", type=str, default=dir_path+"/data/", help="data path")
parser.add_argument('--dataset_str', type=str, default='Musical', help='video, Beauty, Clothing, Music, Musical')
parser.add_argument('--user_feat_type', type=str, default='smooth', help='count, smooth, smooth_imputed')
parser.add_argument("--feat_selection_num", type=int, default=350, help="recommending how many keep at processing")
parser.add_argument('--model_name', type=str, default='NAR', help='[NCF, VBPR, CER; NAR, CAR, CNR]')
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--dim', type=int, default=300, help="Hidden Dimension")
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--use_output_layer", type=int, default=1)
parser.add_argument("--single_output_layer", type=int, default=1)
parser.add_argument('--norm_feat', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
parser.add_argument('--arch_lr', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--opt', type=str, default='Adagrad', help='choice of opt')
parser.add_argument('--batch_size', type=int, default=512, help='choose batch size')
parser.add_argument('--input_dim', type=int, default=2, help='the dimension of input: [2, 3, 4]')
parser.add_argument('--att_mode', type=str, default='0', help='the way of cross-attention: [0, 1, 2, 3]')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

parser.add_argument('--explain_evaluate', type=int, default=1, help='evaluate the explanation performance')
parser.add_argument('--search_epochs', type=int, default=20, help='num of searching epochs')
parser.add_argument('--save', type=str, default='EXP')
parser.add_argument('--save_vis', type=int, default=1)
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--mode', type=str, default='sif-no-auto', help='choose how to search')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
args = parser.parse_args()
save_name = 'experiments/{}/search-Quadruple-{}-{}-{}'.format(args.dataset_str, args.opt, args.arch_lr, time.strftime("%Y%m%d-%H%M%S"))

if args.unrolled:
    save_name += '-unrolled'
utils.create_exp_dir(save_name, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save_name, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def run(args, data, model, evaluator, logging):
	best_ndcg, best_interaction, best_test = 0, '', ''

	g, gp = model.genotype()
	pro = gp.tolist()
	logging.info('genotype: %s' % g)
	logging.info('genotype_p: %s, Entropy: %s' % (gp, entropy(pro)))

	optimizer = torch.optim.Adam(model.parameters(), args.lr)
	#arch_optimizer = torch.optim.Adam(model.arch_parameters(), args.arch_lr)
	if args.opt == 'Adagrad':
		arch_optimizer = torch.optim.Adagrad(model.arch_parameters(), args.arch_lr)
	elif args.opt == 'Adam':
		arch_optimizer = torch.optim.Adam(model.arch_parameters(), args.arch_lr)
	elif args.opt == 'SGD':
		arch_optimizer = torch.optim.SGD(model.arch_parameters(), lr=args.arch_lr, weight_decay=args.weight_decay)

	for search_epoch in range(args.search_epochs):
		g, gp, loss = train_nar_ce(data, model, optimizer, arch_optimizer, args)

		if search_epoch % 5 == 0:
			model.binarize()
			map_train, mrr_train, p_train, r_train, f1_train, hit_train, ndcg_train = test_training_ce(data, model, evaluator)
			test_loss, map, mrr, p, r, f1, hit, ndcg = test_nar_ce(data, model, evaluator)
			model.recover()

			logging.info('search_epoch: %d, training loss: %.4f, test loss: %.4f' % (search_epoch, loss, test_loss))
			logging.info('Performance on training set, map: %.4f, mrr: %.4f, p: %.4f, r: %.4f, f1: %.4f, hit: %.4f, ndcg: %.4f' % (
				map_train, mrr_train, p_train, r_train, f1_train, hit_train, ndcg_train))
			logging.info('Performance on test set: map: %.4f, mrr: %.4f, p: %.4f, r: %.4f, f1: %.4f, hit: %.4f, ndcg: %.4f' % (map, mrr, p, r, f1, hit, ndcg))
			logging.info('genotype: %s' % g)
			logging.info('genotype_p: %s' % (gp))

			if ndcg > best_ndcg:
				best_ndcg = ndcg
				best_interaction = g
				best_test = 'p:{:#.4g} & r:{:#.4g} & f1:{:#.4g} & hit:{:#.4g} & ndcg:{:#.4g} & mrr:{:#.4g}'.format(p, r, f1, hit, ndcg, mrr)
				torch.save(model.state_dict(), os.path.join(save_name, 'model.pt'))

	return best_ndcg, best_interaction, best_test


def main():
	torch.set_default_tensor_type(torch.FloatTensor)
	torch.set_num_threads(3)
	if not torch.cuda.is_available():
		logging.info('no gpu device available')
		sys.exit(1)
	
	np.random.seed(args.seed)
	torch.cuda.set_device(args.gpu)
	cudnn.benchmark = True
	torch.manual_seed(args.seed)
	cudnn.enabled = True
	torch.cuda.manual_seed(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.device = device

	logging.info('gpu device = %d' % args.gpu)
	logging.info("args = %s", args)
	
	data_start = time.time()
	data = DataLoader(args)
	evaluator = Evaluate(args.K)
	logging.info('prepare data finish! [%f]' % (time.time()-data_start))

	if args.explain_evaluate:
		print("loading model for explanation evaluation...")
		att_w = args.att_mode
		model_dim = args.input_dim
		if model_dim == 2:
			model = Network_NAR(data.user_num, data.item_num, args.dim, args.weight_decay, data.w2v_feat_np, data.user_feat_dim, att_w).cuda()
		elif model_dim == 3:
			model = Network_NAR_Triple(data.user_num, data.item_num, args.dim, args.weight_decay, data.w2v_feat_np, data.user_feat_dim, att_w).cuda()
		else:
			model = Network_NAR_Quadruple(data.user_num, data.item_num, args.dim, args.weight_decay, data.w2v_feat_np, data.user_feat_dim, att_w).cuda()

		model_save_path = parent_path+'/ExplainableRec/saved_model/best_{}_{}.pkl'.format(args.model_name, args.dataset_str)
		model.load_state_dict(torch.load(model_save_path))
		model = model.to(args.device)
		vis_func(args, data, model)
		exit(1)

	g_dim, g_best_ndcg, g_best_att, g_best_interaction, g_best_test = 0, 0, 0, '', ''

	inputs = [2, 3, 4]
	atten = [0, 1, 2, 3]
	for dim in inputs:
		search_start = time.time()
		logging.info("\n\n\n\n")
		if dim == 2:
			for att_w in atten:
				logging.info("\n\n")
				logging.info("dim: {}, att_w: {}".format(dim, att_w))
				model = Network_NAR(data.user_num, data.item_num, args.dim, args.weight_decay, data.w2v_feat_np, data.user_feat_dim, att_w).cuda()
				best_ndcg, best_interaction, best_test = run(args, data, model, evaluator, logging)
				logging.info("Best performance: {} with {} for att:{}".format(best_test, best_interaction, att_w))
				if best_ndcg > g_best_ndcg:
					g_best_ndcg = best_ndcg
					g_best_att = att_w
					g_best_interaction = best_interaction
					g_best_test = best_test
					g_dim = 2
					# torch.save(model.state_dict(), os.path.join(save_name, 'model.pt'))
					save_model_func(args, model)

					if args.save_vis:
						vis_func(args, data, model)
		elif dim == 3:
			u_i = [0, 1]
			u_i_atten = []
			for i in u_i:
				for j in atten:
					u_i_atten.append(str(i) + "-" + str(j))
			#u_i_atten=atten
			# for explanation
			for att_w in u_i_atten:
				logging.info("\n\n")
				logging.info("dim: {}, att_w: {}".format(dim, att_w))
				model = Network_NAR_Triple(data.user_num, data.item_num, args.dim, args.weight_decay, data.w2v_feat_np, data.user_feat_dim, att_w).cuda()
				best_ndcg, best_interaction, best_test = run(args, data, model, evaluator, logging)
				logging.info("Best performance: {} with {} for att:{}".format(best_test, best_interaction, att_w))
				if best_ndcg > g_best_ndcg:
					g_best_ndcg = best_ndcg
					g_best_att = att_w
					g_best_interaction = best_interaction
					g_best_test = best_test
					g_dim = 3
					# torch.save(model.state_dict(), os.path.join(save_name, 'model.pt'))
					save_model_func(args, model)

					if args.save_vis:
						vis_func(args, data, model)
		elif dim == 4:
			for att_w in atten:
				logging.info("\n\n")
				logging.info("dim: {}, att_w: {}".format(dim, att_w))
				model = Network_NAR_Quadruple(data.user_num, data.item_num, args.dim, args.weight_decay, data.w2v_feat_np, data.user_feat_dim, att_w).cuda()
				best_ndcg, best_interaction, best_test = run(args, data, model, evaluator, logging)
				logging.info("Best performance: {} with {} for att:{}".format(best_test, best_interaction, att_w))
				if best_ndcg > g_best_ndcg:
					g_best_ndcg = best_ndcg
					g_best_att = att_w
					g_best_interaction = best_interaction
					g_best_test = best_test
					g_dim = 4
					# torch.save(model.state_dict(), os.path.join(save_name, 'model.pt'))
					save_model_func(args, model)

					if args.save_vis:
						vis_func(args, data, model)
		logging.info("\n\n")
		logging.info("Global Best performance: {} with inputs: {}".format(g_best_test, g_dim))
		logging.info("The way of attention: {}, and interaction: {}".format(g_best_att, g_best_interaction))

if __name__ == '__main__':
    main()



