import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
from models_binary import OPS, constrain
from collections import defaultdict
'''
same dimension
p+q --> p+q+r, p+q*r, q+p*r, (p+q)*r, [p+q, r]
p*q --> p*q*r, p*q+r, [p*q, r]
[p,q] --> [p, q, r]

different dimension
[p+q, r]
[p*q, r]
[p, q, r]
'''
PRIMITIVES_TRIPLE = ['plus_concat', 'multiply_concat', 'concat_concat']
PRIMITIVES_NAS = [0, 2, 4]

def ops_triple(triple, p, q, r):
	if triple == 'plus_concat':
		return OPS['concat'](OPS['plus'](p, q), r)
	elif triple == 'multiply_concat':
		return OPS['concat'](OPS['multiply'](p, q), r)
	elif triple == 'concat_concat':
		return OPS['concat'](OPS['concat'](p, q), r)

def _concat(xs):
	return torch.cat([x.view(-1) for x in xs])


def MixedTriple(embedding_p, embedding_q, embedding_r, weights, FC):
	return torch.sum(torch.stack([w * fc(ops_triple(primitive, embedding_p, embedding_q, embedding_r)) \
		for w,primitive,fc in zip(weights,PRIMITIVES_TRIPLE,FC)]), 0)


class Virtue_Triple(nn.Module):
	def __init__(self, num_users, num_items, embedding_dim, reg, w2v, feat_dim, att):
		super(Virtue_Triple, self).__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.embedding_dim = embedding_dim
		self.reg = reg
		self.w2v = w2v
		self.feat_dim = feat_dim
		self.att = att
		self._UsersEmbedding = nn.Embedding(num_users, embedding_dim)
		self._ItemsEmbedding = nn.Embedding(num_items, embedding_dim)

	def compute_loss(self, user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch, neg_item_batch, neg_item_feature_batch):
		# compute positive socre
		pos_score = self.forward(user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch)
		# compute negative socre
		neg_score = self.forward(user_batch, user_feature_batch, neg_item_batch, neg_item_feature_batch)
		# compute loss
		loss = - torch.nn.functional.logsigmoid(pos_score - neg_score).mean()

		#print ("Logits: ", (pos_score - neg_score)[0])
		return loss
	

class NCF_Triple(Virtue_Triple):
	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(NCF_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)
		self._W = nn.Linear(3*embedding_dim, embedding_dim)

	def forward(self, ps, qs, rs):
		constrain(next(self._FC.parameters()))
		constrain(next(self._W.parameters()))

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		gmf_out = ps_embedding * qs_embedding * rs_embedding
		mlp_out = self._W(torch.cat([ps_embedding, qs_embedding, rs_embedding], dim=-1))
		inferences = self._FC(F.relu(gmf_out + mlp_out))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs


class DeepWide_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(DeepWide_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.Linear(3*embedding_dim, 1, bias=False)

	def forward(self, ps, qs, rs):
		constrain(next(self._FC.parameters()))

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		inferences = self._FC(torch.cat([ps_embedding, qs_embedding, rs_embedding], dim=-1))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs


class CP(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(CP, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, ps, qs, rs):
		constrain(next(self._FC.parameters()))

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		inferences = self._FC(ps_embedding * qs_embedding * rs_embedding)
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs


class TuckER(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(TuckER, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		w = torch.empty(embedding_dim, embedding_dim, embedding_dim)
		nn.init.xavier_uniform_(w)
		self._W = torch.nn.Parameter(torch.tensor(w, dtype=torch.float, device='cuda', requires_grad=True))

	def forward(self, ps, qs, rs):
		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		W_after_p = torch.mm(ps_embedding, self._W.view(ps_embedding.size(1), -1))
		W_after_p = W_after_p.view(-1, rs_embedding.size(1), qs_embedding.size(1))
		W_after_r = torch.bmm(rs_embedding.view(-1,1,rs_embedding.size(1)), W_after_p)
		W_after_q = torch.bmm(W_after_r, qs_embedding.view(-1,qs_embedding.size(1),1))
		inferences = W_after_q.view(-1,1)
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs


class NAS_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, arch, reg):
		super(NAS_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = []

		for i in range(len(arch)):
			if i == 0:
				self._FC.append(nn.Linear(3*embedding_dim, int(arch[i])))
			else:
				self._FC.append(nn.Linear(int(arch[i-1]), int(arch[i])))
			self._FC.append(nn.ReLU())
		if len(self._FC) == 0:
			self._FC.append(nn.Linear(3*embedding_dim, 1, bias=False))
		else:
			self._FC.append(nn.Linear(arch[-1], 1, bias=False))
		self._FC = nn.Sequential(*self._FC)

	def forward(self, ps, qs, rs):
		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		inferences = self._FC(torch.cat([ps_embedding, qs_embedding, rs_embedding], dim=-1))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs
	

class AutoNeural_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(AutoNeural_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.Sequential(
			nn.Linear(3*embedding_dim, 3*embedding_dim),
			nn.Sigmoid(),
			nn.Linear(3*embedding_dim, 1))

	def forward(self, ps, qs, rs):
		for p in self._FC.parameters():
			if len(p.size()) == 1: continue
			constrain(p)

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		inferences = self._FC(torch.cat([ps_embedding,qs_embedding,rs_embedding], dim=-1))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))

		return inferences, regs

	def embedding_parameters(self):
		return list(self._PsEmbedding.parameters()) + list(self._QsEmbedding.parameters()) + \
			list(self._RsEmbedding.parameters())

	def mlp_parameters(self):
		return self._FC.parameters()


class Network_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, arch, reg):
		super(Network_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self.arch = arch
		self.mlp_p = arch['mlp']['p']
		self.mlp_q = arch['mlp']['q']
		self.mlp_r = arch['mlp']['r']

		if arch['triple'] == 'concat_concat':
			self._FC = nn.Linear(3*embedding_dim, 1, bias=False)
		elif 'concat' in arch['triple']:
			self._FC = nn.Linear(2*embedding_dim, 1, bias=False)
		else:
			self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def parameters(self):
		return list(self._PsEmbedding.parameters()) + list(self._QsEmbedding.parameters()) + \
			list(self._RsEmbedding.parameters()) + list(self._FC.parameters())

	def forward(self, ps, qs, rs):
		constrain(next(self._FC.parameters()))
		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		ps_embedding_trans = self.mlp_p(ps_embedding.view(-1,1)).view(ps_embedding.size())
		qs_embedding_trans = self.mlp_q(qs_embedding.view(-1,1)).view(qs_embedding.size())
		rs_embedding_trans = self.mlp_r(rs_embedding.view(-1,1)).view(rs_embedding.size())

		inferences = self._FC(ops_triple(self.arch['triple'], ps_embedding_trans, 
			qs_embedding_trans, rs_embedding_trans))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + \
			torch.norm(rs_embedding))
		return inferences, regs


class Network_NAR_Triple(Virtue_Triple):
	def __init__(self, num_users, num_items, embedding_dim, reg, w2v, feat_dim, att):
		super(Network_NAR_Triple, self).__init__(num_users, num_items, embedding_dim, reg, w2v, feat_dim, att)
		self.num_users = num_users
		self.num_items = num_items
		self.embedding_dim = embedding_dim
		self.reg = reg
		self.w2v = w2v
		self.feat_dim = feat_dim
		self.att = att
		self.atten_fusion = nn.Linear(feat_dim*2, feat_dim, bias=False)

		self.word_feat = torch.FloatTensor(w2v).cuda()
		self.word_feat = self.word_feat.cuda()
		output_hidden_dim = embedding_dim+w2v.shape[1]
		self.output_linear = nn.Linear(output_hidden_dim, 1)
		self.drop = nn.Dropout(p=0.5)

		self._FC = nn.ModuleList()
		for primitive in PRIMITIVES_TRIPLE:
			if primitive == 'concat_concat':
				self._FC.append(nn.Linear(3*embedding_dim, 1, bias=False))
			elif 'concat' in primitive:
				self._FC.append(nn.Linear(2*embedding_dim, 1, bias=False))
			else:
				self._FC.append(nn.Linear(embedding_dim, 1, bias=False))
		self._initialize_alphas()

	def _initialize_alphas(self):
		nn.init.normal_(self._UsersEmbedding.weight, std=0.01)
		nn.init.normal_(self._ItemsEmbedding.weight, std=0.01)
		self._UsersEmbedding.weight.requires_grad = True
		self._ItemsEmbedding.weight.requires_grad = True

		self.mlp_p = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
		self.mlp_q = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

		self._arch_parameters = {}
		self._arch_parameters['mlp'] = {}
		self._arch_parameters['mlp']['p'] = self.mlp_p
		self._arch_parameters['mlp']['q'] = self.mlp_q

		self._arch_parameters['triple'] = Variable(torch.ones(len(PRIMITIVES_TRIPLE), dtype=torch.float, device='cuda') /2, requires_grad=True)
		self._arch_parameters['triple'].data.add_(
			torch.randn_like(self._arch_parameters['triple'])*1e-3)


	def arch_parameters(self):
		return list(self._arch_parameters['mlp']['p'].parameters()) + \
			   list(self._arch_parameters['mlp']['q'].parameters()) + \
			   [self._arch_parameters['triple']]

	def new(self):
		model_new = Network_NAR_Triple(self.num_users, self.num_items, self.embedding_dim, self.reg, self.w2v, self.feat_dim, self.att).cuda()
		for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
			x.data = y.data.clone()
			try:
				x.grad = y.grad.clone()
			except:
				pass
		return model_new

	def clip(self):
		m = nn.Hardtanh(0, 1)
		self._arch_parameters['triple'].data = m(self._arch_parameters['triple'])
	
	def binarize(self):
		self._cache = self._arch_parameters['triple'].clone()
		max_index = self._arch_parameters['triple'].argmax().item()
		for i in range(self._arch_parameters['triple'].size(0)):
			if i == max_index:
				self._arch_parameters['triple'].data[i] = 1.0
			else:
				self._arch_parameters['triple'].data[i] = 0.0
	
	def recover(self):
		self._arch_parameters['triple'].data = self._cache
		del self._cache

	def forward(self, user, user_feat, item, item_feat):
		user_feat = F.normalize(user_feat, p=2, dim=1)
		item_feat = F.normalize(item_feat, p=2, dim=1)

		for i in range(len(PRIMITIVES_TRIPLE)):
			constrain(next(self._FC[i].parameters()))

		users_embedding = self._UsersEmbedding(user)
		items_embedding = self._ItemsEmbedding(item)
		users_embedding_trans = self._arch_parameters['mlp']['p'](users_embedding)
		items_embedding_trans = self._arch_parameters['mlp']['q'](items_embedding)
		user_word_feat_att = torch.mm(users_embedding_trans, self.word_feat.t())
		item_word_feat_att = torch.mm(items_embedding_trans, self.word_feat.t())

		u_i, attention = self.att.split('-')
		if attention == '0':
			user_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, user_feat], dim=-1)))
			item_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, item_feat], dim=-1)))
		elif attention == '1':
			user_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, user_feat], dim=-1)))
			item_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, item_feat], dim=-1)))
		elif attention == '2':
			user_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, user_feat], dim=-1)))
			item_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, item_feat], dim=-1)))
		elif attention == '3':
			user_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, user_feat], dim=-1)))
			item_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, item_feat], dim=-1)))

		user_word_embedding = torch.matmul(user_fused_att,self.word_feat)
		item_word_embedding = torch.matmul(item_fused_att,self.word_feat)

		user_word_embedding = self.drop(user_word_embedding)
		item_word_embedding = self.drop(item_word_embedding)

		if u_i == '0':
			output = MixedTriple(user_word_embedding, item_word_embedding, users_embedding,
								 self._arch_parameters['triple'], self._FC)
		else:
			output = MixedTriple(user_word_embedding, item_word_embedding, items_embedding,
								 self._arch_parameters['triple'], self._FC)
		return output

	def vis(self, data):
		train_users = data.train_user_all
		pos_items = data.pos_item_all
		print("the number of pos items:{}, distinct number:{}".format(len(pos_items), len(set(pos_items.tolist()))))

		user_feat = data.user_feature_all[train_users]
		item_feat = data.item_feature_all[pos_items]

		user_feat = F.normalize(user_feat, p=2, dim=1)
		item_feat = F.normalize(item_feat, p=2, dim=1)

		train_users = torch.tensor(train_users).cuda()
		user_embedding = self._UsersEmbedding(train_users)
		user_embedding = self._arch_parameters['mlp']['p'](user_embedding)
		user_word_feat_att = torch.mm(user_embedding, self.word_feat.t())

		pos_items = torch.tensor(pos_items).cuda()
		item_embedding = self._ItemsEmbedding(pos_items)
		item_embedding = self._arch_parameters['mlp']['q'](item_embedding)
		item_word_feat_att = torch.mm(item_embedding, self.word_feat.t())

		u_i, attention = self.att.split('-')
		if attention == '0':
			user_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, user_feat], dim=-1))).detach().cpu().numpy()
			item_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, item_feat], dim=-1))).detach().cpu().numpy()
		elif attention == '1':
			user_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, user_feat], dim=-1))).detach().cpu().numpy()
			item_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, item_feat], dim=-1))).detach().cpu().numpy()
		elif attention == '2':
			user_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, user_feat], dim=-1))).detach().cpu().numpy()
			item_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, item_feat], dim=-1))).detach().cpu().numpy()
		elif attention == '3':
			user_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, user_feat], dim=-1))).detach().cpu().numpy()
			item_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, item_feat], dim=-1))).detach().cpu().numpy()

		user_att_dict = defaultdict(list)
		for idx, user in enumerate(train_users):
			user_att_dict[user.item()] = user_fused_att[idx]

		item_att_dict = defaultdict(list)
		for idx, item in enumerate(pos_items):
			item_att_dict[item.item()] = item_fused_att[idx]
			#if item.item() == 6:
			#print("item_att_dict:{}".format(item_fused_att[idx]))
		print("the number of items in the attention matrix:{}".format(len(item_att_dict)))
		return user_att_dict, item_att_dict


	def genotype(self):
		genotype = PRIMITIVES_TRIPLE[self._arch_parameters['triple'].argmax().cpu().numpy()]
		genotype_p = F.softmax(self._arch_parameters['triple'], dim=-1)
		return genotype, genotype_p.cpu().detach()


	def step(self, users_train, users_feat_train, pos_items_train, pos_items_feat_train, neg_items_train, neg_items_feat_train,
			 users_valid, users_feat_valid, pos_items_valid, pos_items_feat_valid, neg_items_valid, neg_items_feat_valid,
			 lr, arch_optimizer, unrolled):
		self.zero_grad()
		arch_optimizer.zero_grad()

		# binarize before forward propagation
		self.binarize()
		if unrolled:
			loss = self._backward_step_unrolled(users_train, users_feat_train, pos_items_train, pos_items_feat_train, neg_items_train, neg_items_feat_train,
												users_valid, users_feat_valid, pos_items_valid, pos_items_feat_valid, neg_items_valid, neg_items_feat_valid, lr)
		else:
			loss = self._backward_step(users_valid, users_feat_valid, pos_items_valid, pos_items_feat_valid, neg_items_valid, neg_items_feat_valid)
		# restore weight before updating
		self.recover()
		arch_optimizer.step()
		return loss


	def _backward_step(self, users_valid, users_feat_valid, pos_items_valid, pos_items_feat_valid, neg_items_valid, neg_items_feat_valid):
		loss = self.compute_loss(users_valid, users_feat_valid, pos_items_valid, pos_items_feat_valid, neg_items_valid, neg_items_feat_valid)
		loss.backward()
		return loss


	def _backward_step_unrolled(self, users_train, users_feat_train, pos_items_train, pos_items_feat_train, neg_items_train, neg_items_feat_train,
								users_valid, users_feat_valid, pos_items_valid, pos_items_feat_valid, neg_items_valid, neg_items_feat_valid, lr):
		unrolled_model = self._compute_unrolled_model(
			users_train, users_feat_train, pos_items_train, pos_items_feat_train, neg_items_train, neg_items_feat_train, lr)
		unrolled_loss = unrolled_model.compute_loss(users_valid, users_feat_valid, pos_items_valid, pos_items_feat_valid, neg_items_valid, neg_items_feat_valid)

		unrolled_loss.backward()
		dalpha = [v.grad for v in unrolled_model.arch_parameters()]
		vector = [v.grad for v in unrolled_model.parameters()]
		implicit_grads = self._hessian_vector_product(vector, users_train, users_feat_train, pos_items_train, pos_items_feat_train, neg_items_train, neg_items_feat_train)

		for g,ig in zip(dalpha,implicit_grads):
			g.sub_(lr, ig)

		for v,g in zip(self.arch_parameters(), dalpha):
			v.grad = g.clone()
		return unrolled_loss

	def _compute_unrolled_model(self, users_train, users_feat_train, pos_items_train, pos_items_feat_train, neg_items_train, neg_items_feat_train, lr):
		#inferences, regs = self(users_train, users_feat_valid, items_train, items_feat_train,)
		#loss = self.compute_loss(inferences, labels_train, regs)
		loss = self.compute_loss(self, users_train, users_feat_train, pos_items_train, pos_items_feat_train, neg_items_train, neg_items_feat_train)

		theta = _concat(self.parameters())
		dtheta = _concat(torch.autograd.grad(loss, self.parameters())) + self.reg * theta
		unrolled_model = self._construct_model_from_theta(theta.sub(lr, dtheta))
		return unrolled_model

	def _construct_model_from_theta(self, theta):
		model_new = self.new()
		model_dict = self.state_dict()
		params, offset = {}, 0
		for k,v in self.named_parameters():
			v_length = np.prod(v.size())
			params[k] = theta[offset: offset+v_length].view(v.size())
			offset += v_length

		assert offset == len(theta)
		model_dict.update(params)
		model_new.load_state_dict(model_dict)
		return model_new.cuda()

	def _hessian_vector_product(self, vector, p_train, q_train, r_train, labels_train, r=1e-2):
		R = r / _concat(vector).norm()
		for p,v in zip(self.parameters(), vector):
			p.data.add_(R, v)
		inferences, regs = self(p_train, q_train, r_train)
		loss = self.compute_loss(inferences, labels_train, regs)
		grads_p = torch.autograd.grad(loss, self.arch_parameters())

		for p,v in zip(self.parameters(), vector):
			p.data.sub_(2*R, v)
		inferences, regs = self(p_train, q_train, r_train)
		loss = self.compute_loss(inferences, labels_train, regs)
		grads_n = torch.autograd.grad(loss, self.arch_parameters())

		for p,v in zip(self.parameters(), vector):
			p.data.add_(R, v)

		return [(x-y).div_(2*R) for x,y in zip(grads_p,grads_n)]
