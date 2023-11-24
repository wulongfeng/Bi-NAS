import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from collections import defaultdict


PRIMITIVES_BINARY = ['plus', 'multiply', 'concat']
PRIMITIVES_NAS = [0, 2, 4]
SPACE_NAS = pow(len(PRIMITIVES_NAS), 3)


OPS = {
	'plus': lambda p, q: p + q,
	'multiply': lambda p, q: p * q,
	'concat': lambda p, q: torch.cat([p, q], dim=-1),
	'norm_0': lambda p: torch.ones_like(p),
	'norm_0.5': lambda p: torch.sqrt(torch.abs(p) + 1e-7),
	'norm_1': lambda p: torch.abs(p),
	'norm_2': lambda p: p ** 2,
	'I': lambda p: torch.ones_like(p),
	'-I': lambda p: -torch.ones_like(p),
	'sign': lambda p: torch.sign(p),
}


def constrain(p):
	c = torch.norm(p, p=2, dim=1, keepdim=True)
	c[c < 1] = 1.0
	p.data.div_(c)


def MixedBinary(embedding_p, embedding_q, weights, FC):
	return torch.sum(torch.stack([w * fc(OPS[primitive](embedding_p, embedding_q)) \
		for w,primitive,fc in zip(weights,PRIMITIVES_BINARY,FC)]), 0)


def MultiplyLoss(embedding_p, embedding_q, FC):
	fc = FC[0]
	primitive = PRIMITIVES_BINARY[0]
	return fc(OPS[primitive](embedding_p, embedding_q))

def _concat(xs):
	return torch.cat([x.view(-1) for x in xs])


class Virtue(nn.Module):
	def __init__(self, num_users, num_items, embedding_dim, reg, w2v, feat_dim, att):
		super(Virtue, self).__init__()
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


class Network(Virtue):
    def __init__(self, num_users, num_items, embedding_dim, arch, reg, w2v, feat_dim, att):
        super(Network, self).__init__(num_users, num_items, embedding_dim, reg, w2v, feat_dim, att)
        self.arch = arch
        self.mlp_p = arch['mlp']['p']
        self.mlp_q = arch['mlp']['q']
        
        if arch['binary'] == 'concat':
            self._FC = nn.Linear(2*embedding_dim, 1, bias=False)
        else:
            self._FC = nn.Linear(embedding_dim, 1, bias=False)
    
    def parameters(self):
        return list(self._UsersEmbedding.parameters()) + list(self._ItemsEmbedding.parameters()) + \
            list(self._FC.parameters())
    
    def forward(self, users, items):
        constrain(next(self._FC.parameters()))
        users_embedding = self._UsersEmbedding(users)
        items_embedding = self._ItemsEmbedding(items)
        
        users_embedding_trans = self.mlp_p(users_embedding.view(-1,1)).view(users_embedding.size())
        items_embedding_trans = self.mlp_q(items_embedding.view(-1,1)).view(items_embedding.size())
        
        inferences = self._FC(OPS[self.arch['binary']](users_embedding_trans, items_embedding_trans))
        regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
        return inferences, regs


class Network_Search(Virtue):
    def __init__(self, num_users, num_items, embedding_dim, reg, w2v, feat_dim, att):
        super(Network_Search, self).__init__(num_users, num_items, embedding_dim, reg, w2v, feat_dim, att)
        self._FC = nn.ModuleList()
        for primitive in PRIMITIVES_BINARY:
            if primitive == 'concat':
                self._FC.append(nn.Linear(2*embedding_dim, 1, bias=False))
            else:
                self._FC.append(nn.Linear(embedding_dim, 1, bias=False))
        self._initialize_alphas()
    
    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
			nn.Linear(1, 8),
			nn.Tanh(),
			nn.Linear(8, 1)).cuda()
        self.mlp_q = nn.Sequential(
			nn.Linear(1, 8),
			nn.Tanh(),
            nn.Linear(8, 1)).cuda()
        self._arch_parameters = {}
        self._arch_parameters['mlp'] = {}
        self._arch_parameters['mlp']['p'] = self.mlp_p
        self._arch_parameters['mlp']['q'] = self.mlp_q
        self._arch_parameters['binary'] = Variable(torch.ones(len(PRIMITIVES_BINARY), 
            dtype=torch.float, device='cuda') / 2, requires_grad=True)
        self._arch_parameters['binary'].data.add_(
            torch.randn_like(self._arch_parameters['binary'])*1e-3)
    
    def arch_parameters(self):
        return list(self._arch_parameters['mlp']['p'].parameters()) + \
            list(self._arch_parameters['mlp']['q'].parameters()) + [self._arch_parameters['binary']]
    
    def new(self):
        model_new = Network_Search(self.num_users, self.num_items, self.embedding_dim, self.reg).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data = y.data.clone()
        return model_new
    
    def clip(self):
        m = nn.Hardtanh(0, 1)
        self._arch_parameters['binary'].data = m(self._arch_parameters['binary'])
    
    def binarize(self):
        self._cache = self._arch_parameters['binary'].clone()
        max_index = self._arch_parameters['binary'].argmax().item()
        for i in range(self._arch_parameters['binary'].size(0)):
            if i == max_index:
                self._arch_parameters['binary'].data[i] = 1.0
            else:
                self._arch_parameters['binary'].data[i] = 0.0
    
    def recover(self):
        self._arch_parameters['binary'].data = self._cache
        del self._cache

    def forward(self, users, items):
        for i in range(len(PRIMITIVES_BINARY)):
            constrain(next(self._FC[i].parameters()))

        users_embedding = self._UsersEmbedding(users)
        items_embedding = self._ItemsEmbedding(items)

        users_embedding_trans = self._arch_parameters['mlp']['p'](users_embedding.view(-1,1)).view(users_embedding.size())
        items_embedding_trans = self._arch_parameters['mlp']['q'](items_embedding.view(-1,1)).view(items_embedding.size())

        # the weight is already binarized
        assert self._arch_parameters['binary'].sum() == 1.
        inferences = MixedBinary(users_embedding_trans, items_embedding_trans,
                                 self._arch_parameters['binary'], self._FC)

        regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
        return inferences, regs

    def genotype(self):
        genotype = PRIMITIVES_BINARY[self._arch_parameters['binary'].argmax().cpu().numpy()]
        genotype_p = F.softmax(self._arch_parameters['binary'], dim=-1)
        return genotype, genotype_p.cpu().detach()

    def step(self, users_train, items_train, labels_train, users_valid, 
		items_valid, labels_valid, lr, arch_optimizer, unrolled):
        self.zero_grad()
        arch_optimizer.zero_grad()

        # binarize before forward propagation
        self.binarize()
        if unrolled:
            loss = self._backward_step_unrolled(users_train, items_train, labels_train,
				users_valid, items_valid, labels_valid, lr)
        else:
            loss = self._backward_step(users_valid, items_valid, labels_valid)
        # restore weight before updating
        self.recover()
        arch_optimizer.step()
        return loss
    
    def _backward_step(self, users_valid, items_valid, labels_valid):
        inferences, regs = self(users_valid, items_valid)
        loss = self.compute_loss(inferences, labels_valid, regs)
        loss.backward()
        return loss
    
    def _backward_step_unrolled(self, users_train, items_train, labels_train,
		users_valid, items_valid, labels_valid, lr):
        unrolled_model = self._compute_unrolled_model(
			users_train, items_train, labels_train, lr)
        unrolled_inference, unrolled_regs = unrolled_model(users_valid, items_valid)
        unrolled_loss = unrolled_model.compute_loss(unrolled_inference, labels_valid, unrolled_regs)
        
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, users_train, items_train, labels_train)
        
        for g,ig in zip(dalpha,implicit_grads):
            g.sub_(lr, ig)
        
        for v,g in zip(self.arch_parameters(), dalpha):
            v.grad = g.clone()
        return unrolled_loss
    
    def _compute_unrolled_model(self, users_train, items_train, labels_train, lr):
        inferences, regs = self(users_train, items_train)
        loss = self.compute_loss(inferences, labels_train, regs)
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
    
    def _hessian_vector_product(self, vector, users, items, labels, r=1e-2):
        R = r / _concat(vector).norm()
        for p,v in zip(self.parameters(), vector):
            p.data.add_(R, v)
        inferences, regs = self(users, items)
        loss = self.compute_loss(inferences, labels, regs)
        grads_p = torch.autograd.grad(loss, self.arch_parameters())

        for p,v in zip(self.parameters(), vector):
            p.data.sub_(2*R, v)
        inferences, regs = self(users, items)
        loss = self.compute_loss(inferences, labels, regs)
        grads_n = torch.autograd.grad(loss, self.arch_parameters())

        for p,v in zip(self.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x,y in zip(grads_p,grads_n)]


class Network_NAR(Virtue):
	def __init__(self, num_users, num_items, embedding_dim, reg, w2v, feat_dim, att):
		super(Network_NAR, self).__init__(num_users, num_items, embedding_dim, reg, w2v, feat_dim, att)
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
		for primitive in PRIMITIVES_BINARY:
			if primitive == 'concat':
				self._FC.append(nn.Linear(2*embedding_dim, 1, bias=False))
			else:
				#self._FC.append(nn.Linear(embedding_dim, 1, bias=False))
				self._FC.append(nn.Linear(embedding_dim, 1))
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
		self._arch_parameters['binary'] = Variable(torch.ones(len(PRIMITIVES_BINARY), dtype=torch.float, device='cuda'), requires_grad=True)
		self._arch_parameters['binary'].data.add_(
			torch.randn_like(self._arch_parameters['binary'])*1e-3)

	def arch_parameters(self):
		return list(self._arch_parameters['mlp']['p'].parameters()) + \
			   list(self._arch_parameters['mlp']['q'].parameters()) + [self._arch_parameters['binary']]

	def new(self):
		model_new = Network_NAR(self.num_users, self.num_items, self.embedding_dim, self.reg, self.w2v_feat, self.feat_dim, self.att).cuda()
		for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
			x.data = y.data.clone()
		return model_new

	def clip(self):
		m = nn.Hardtanh(0, 1)
		self._arch_parameters['binary'].data = m(self._arch_parameters['binary'])

	def binarize(self):
		self._cache = self._arch_parameters['binary'].clone()
		max_index = self._arch_parameters['binary'].argmax().item()
		for i in range(self._arch_parameters['binary'].size(0)):
			if i == max_index:
				self._arch_parameters['binary'].data[i] = 1.0
			else:
				self._arch_parameters['binary'].data[i] = 0.0

	def recover(self):
		self._arch_parameters['binary'].data = self._cache
		del self._cache

	def forward(self, user, user_feat, item, item_feat):
		#if self.args.norm_feat:
		user_feat = F.normalize(user_feat, p=2, dim=1)
		item_feat = F.normalize(item_feat, p=2, dim=1)

		for i in range(len(PRIMITIVES_BINARY)):
			constrain(next(self._FC[i].parameters()))

		users_embedding = self._UsersEmbedding(user)
		items_embedding = self._ItemsEmbedding(item)

		users_embedding_trans = self._arch_parameters['mlp']['p'](users_embedding)
		items_embedding_trans = self._arch_parameters['mlp']['q'](items_embedding)

		user_word_feat_att = torch.mm(users_embedding_trans, self.word_feat.t())
		item_word_feat_att = torch.mm(items_embedding_trans, self.word_feat.t())

		if self.att == 0:
			user_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, user_feat], dim=-1)))
			item_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, item_feat], dim=-1)))
		elif self.att == 1:
			user_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, user_feat], dim=-1)))
			item_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, item_feat], dim=-1)))
		elif self.att == 2:
			user_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, user_feat], dim=-1)))
			item_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, item_feat], dim=-1)))
		elif self.att == 3:
			user_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, user_feat], dim=-1)))
			item_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, item_feat], dim=-1)))

		user_word_embedding = torch.matmul(user_fused_att, self.word_feat)
		item_word_embedding = torch.matmul(item_fused_att, self.word_feat)

		# the weight is already binarized
		user_word_embedding = self.drop(user_word_embedding)
		item_word_embedding = self.drop(item_word_embedding)

		#print("binary paramters:{}".format(self._arch_parameters['binary']))
		#assert self._arch_parameters['binary'].sum() == 1.
		output = MixedBinary(user_word_embedding, item_word_embedding,
								 self._arch_parameters['binary'], self._FC)
		#output = MultiplyLoss(user_word_embedding, item_word_embedding, self._FC)

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

		if self.att == 0:
			user_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, user_feat], dim=-1))).detach().cpu().numpy()
			item_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, item_feat], dim=-1))).detach().cpu().numpy()
		elif self.att == 1:
			user_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, user_feat], dim=-1))).detach().cpu().numpy()
			item_fused_att = F.relu(self.atten_fusion(torch.cat([user_word_feat_att, item_feat], dim=-1))).detach().cpu().numpy()
		elif self.att == 2:
			user_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, user_feat], dim=-1))).detach().cpu().numpy()
			item_fused_att = F.relu(self.atten_fusion(torch.cat([item_word_feat_att, item_feat], dim=-1))).detach().cpu().numpy()
		elif self.att == 3:
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
		genotype = PRIMITIVES_BINARY[self._arch_parameters['binary'].argmax().cpu().numpy()]
		genotype_p = F.softmax(self._arch_parameters['binary'], dim=-1)
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
		#inferences, regs = self(users_valid, users_feat_valid, items_valid, items_feat_valid)
		#loss = self.compute_loss(inferences, labels_valid, regs)
		loss = self.compute_loss(users_valid, users_feat_valid, pos_items_valid, pos_items_feat_valid, neg_items_valid, neg_items_feat_valid)
		loss.backward()
		return loss

	def _backward_step_unrolled(self, users_train, users_feat_train, pos_items_train, pos_items_feat_train, neg_items_train, neg_items_feat_train,
								users_valid, users_feat_valid, pos_items_valid, pos_items_feat_valid, neg_items_valid, neg_items_feat_valid, lr):
		unrolled_model = self._compute_unrolled_model(
			users_train, users_feat_train, pos_items_train, pos_items_feat_train, neg_items_train, neg_items_feat_train, lr)

		#unrolled_inference, unrolled_regs = unrolled_model(users_valid, users_feat_valid, items_valid, items_feat_valid)
		#unrolled_loss = unrolled_model.compute_loss(unrolled_inference, labels_valid, unrolled_regs)
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

	def _hessian_vector_product(self, vector, users, users_feat, pos_items, pos_items_feat, neg_items, neg_items_feat, r=1e-2):
		R = r / _concat(vector).norm()
		for p,v in zip(self.parameters(), vector):
			p.data.add_(R, v)
		#inferences, regs = self(users, users_feat, items, items_feat)
		#loss = self.compute_loss(inferences, labels, regs)
		loss = self.compute_loss(users, users_feat, pos_items, pos_items_feat, neg_items, neg_items_feat)
		grads_p = torch.autograd.grad(loss, self.arch_parameters())

		for p,v in zip(self.parameters(), vector):
			p.data.sub_(2*R, v)
		#inferences, regs = self(users, users_feat, items, items_feat)
		#loss = self.compute_loss(inferences, labels, regs)
		loss = self.compute_loss(users, users_feat, pos_items, pos_items_feat, neg_items, neg_items_feat)
		grads_n = torch.autograd.grad(loss, self.arch_parameters())

		for p,v in zip(self.parameters(), vector):
			p.data.add_(R, v)

		return [(x-y).div_(2*R) for x,y in zip(grads_p,grads_n)]







