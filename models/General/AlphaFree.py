from sklearn.metrics.pairwise import cosine_similarity
from .base.abstract_model import AbstractModel
from .base.abstract_data import AbstractData
from sklearn.preprocessing import normalize
from .base.abstract_RS import AbstractRS
from scipy import sparse
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import torch

class AlphaFree_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
    
    def save_final_model(self, model, path):
        torch.save(model.state_dict(), path + '/model.pt')
        print("Model saved at", path + '/model.pt')
    
    def train_one_epoch(self, epoch):
        running_loss, num_batches = 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          
            
            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, _, _, neg_items = batch[0], batch[1], batch[2], batch[3], batch[4]
            self.model.train()
            loss = self.model(users, pos_items, neg_items)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.detach().item()
            num_batches += 1
        return [running_loss/num_batches]

class AlphaFree_Data(AbstractData):
    def __init__(self, args):
        super().__init__(args)
    
    def add_special_model_attr(self, args):
        self.lm_model = args.lm_model
        loading_path = args.data_path + args.dataset + '/item_info/'
        embedding_path_dict = {
            'v3': 'item_cf_embeds_large3_array.npy',
            'llama': 'item_cf_embeds_LLAMA_array.npy'
        }
        self.item_cf_embeds = np.load(loading_path + embedding_path_dict[self.lm_model])

        pairs = []
        for u, v in self.train_user_list.items():
            for i in v:
                pairs.append((u, i))
        pairs = pd.DataFrame(pairs, columns=['user_id', 'item_id'])
        self.topk_indices = {}
        
        print("KNN building...")
        self.item_item_sim_dict, self.item_item_sim_matrix = self.build_item_knn_dict(pairs, args.topk_knn+1) # include itself
        print("KNN built!")
        aug_start_time = time.time()
        self.item_cf_embeds2 = self.item_item_sim_matrix.dot(self.item_cf_embeds)

        row = pairs['user_id'].tolist()
        col = pairs['item_id'].tolist()
        val = len(pairs) * [1.0]
        self.user_interact_matrix = sparse.csr_matrix((val, (row, col)), shape=(self.n_users, self.n_items), dtype=np.float32)
        self.user_interact_matrix = normalize(self.user_interact_matrix, norm='l1', axis=1)

        self.aug_user_interact_matrix = self.augment_user_interact_matrix(pairs, self.item_item_sim_dict)
        print("Data preparation done!, time:", time.time() - aug_start_time)
      
        torch.cuda.empty_cache()

    def group_agg(self, group_data, embedding_dict, key='item_id'):
        ids = group_data[key].values
        embeds = [embedding_dict[id] for id in ids]
        embeds = np.array(embeds)
        return embeds.mean(axis=0)

    def build_item_knn_dict(self, user_item_pairs, topk=3, batch_size=1024):
        row = user_item_pairs['user_id'].tolist()
        col = user_item_pairs['item_id'].tolist()
        val = [1.0] * len(user_item_pairs)
        A = sparse.csr_matrix((val, (row, col)), shape=(self.n_users, self.n_items))
        item_item_sim = cosine_similarity(A.T, dense_output=False) 
        item_item_sim = item_item_sim.tocsr()
        print("Calculating embedding-based mean similarity (batch, GPU)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        item_embeds = torch.from_numpy(self.item_cf_embeds).to(device) 

        n_items = item_embeds.shape[0]
        sim_sum = torch.zeros(n_items, device=device)
        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            sims = torch.matmul(item_embeds[start:end], item_embeds.T)  # (batch, n_items)
            sim_sum[start:end] = sims.sum(dim=1)

        embedding_based_sim_mean = (sim_sum / n_items).cpu().numpy()  
        
        print("Finding KNN...")
        item_knn = {}
        for i in range(n_items):
            row_i = item_item_sim.getrow(i)
            indices = row_i.indices
            values = row_i.data

            if len(values) == 0:
                item_knn[i] = []
                continue

            topk_idx = np.argsort(values)[-topk:][::-1]
            
            filtered_neighbors = []
            emb_i = self.item_cf_embeds[i]
            for j in topk_idx:
                neighbor_idx = indices[j]
                sim_val = float(values[j])
                emb_sim = float(np.dot(emb_i, self.item_cf_embeds[neighbor_idx]))
                if emb_sim > embedding_based_sim_mean[i]:
                    filtered_neighbors.append((neighbor_idx, sim_val))
            
            item_knn[i] = filtered_neighbors
            
        row = []
        col = []
        value = []
        for i in item_knn:
            for j, sim in item_knn[i]:
                row.append(i)
                col.append(j)
                value.append(sim)
        item_item_sim = sparse.csr_matrix((value, (row, col)), shape=(self.n_items, self.n_items))
        # no need to symmetrize, already symmetric
        item_item_sim = 0.5 * (item_item_sim + item_item_sim.T) 
        degrees = np.array(item_item_sim.sum(axis=0)).ravel()  #column sum
        deg_inv = np.power(degrees, -1.0)
        deg_inv[np.isinf(deg_inv)] = 0.0
        D_inv = sparse.diags(deg_inv)
        A_hat = item_item_sim @ D_inv
        return item_knn, A_hat

    def group_agg(self, group_data, sim_dict, key='item_id'):
        ids = group_data[key].values
        potential_int_items = set()
        for id in ids:
            temp_inter = sim_dict[id]
            for item in temp_inter:
                potential_int_items.add(item[0])
        return potential_int_items

    def augment_user_interact_matrix(self, pairs, sim_dict):
        groups = pairs.groupby('user_id')
        user_df = groups.apply(self.group_agg, sim_dict, key='item_id')
        row = []
        col = []
        val = []
        for user_id, items in user_df.items():
            for item in items:
                row.append(int(user_id))
                col.append(int(item))
                val.append(1.0)
        
        aug_matrix = sparse.csr_matrix((val, (row, col)), shape=(self.n_users, self.n_items), dtype=np.float32)
        aug_matrix = normalize(aug_matrix, norm='l1', axis=1)
        
        return aug_matrix


class AlphaFree(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau
        self.embed_size = args.hidden_size
        self.lm_model = args.lm_model
        self.init_item_cf_embeds = data.item_cf_embeds
        self.init_item_cf_embeds_aug = data.item_cf_embeds2
        self.align_loss = args.align
        self.align_temperature = args.align_temperature
        self.init_item_cf_embeds = torch.tensor(self.init_item_cf_embeds, dtype=torch.float32).cuda(self.device)
        self.init_item_cf_embeds_aug = torch.tensor(self.init_item_cf_embeds_aug, dtype=torch.float32).cuda(self.device)
        self.user_interact_matrix = data.user_interact_matrix
        self.aug_user_interact_matrix = data.aug_user_interact_matrix
        self.init_embed_shape = self.init_item_cf_embeds.shape[1]

        multiplier_dict = {
            'v3': 1/2,
            'llama': 9/32
        } 
        
        if(self.lm_model in multiplier_dict):
            multiplier = multiplier_dict[self.lm_model]
        else:
            multiplier = 9/32

        self.mlp_origin = nn.Sequential(
            nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
            nn.LeakyReLU(),
            nn.Linear(int(multiplier * self.init_embed_shape), self.embed_size)
        )
        self.mlp_aug = nn.Sequential(
            nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
            nn.LeakyReLU(),
            nn.Linear(int(multiplier * self.init_embed_shape), self.embed_size)
        )

    def compute_origin(self, users):
        user_cf_emb = self.calculate_user_embedding(users)
        users_cf_emb = self.mlp_origin(user_cf_emb)
        items_cf_emb = self.mlp_origin(self.init_item_cf_embeds)

        return users_cf_emb, items_cf_emb

    def compute_aug(self, users):
        user_cf_emb = self.calculate_user_embedding_aug(users)
        users_cf_emb = self.mlp_aug(user_cf_emb)
        items_cf_emb = self.mlp_aug(self.init_item_cf_embeds_aug)
        return users_cf_emb, items_cf_emb
    
    def csr_to_torch_sparse(self, csr_mat):
        csr_mat = csr_mat.tocoo() 
        row = torch.from_numpy(csr_mat.row).long()
        col = torch.from_numpy(csr_mat.col).long()
        indices = torch.stack([row, col], dim=0)
        values = torch.from_numpy(csr_mat.data).float()
        shape = torch.Size(csr_mat.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def calculate_user_embedding_aug(self, users):
        user_interaction_matrix = self.aug_user_interact_matrix[users]
        user_interaction_matrix = self.csr_to_torch_sparse(user_interaction_matrix).to(self.device)
        estimated_user_embeds = torch.sparse.mm(user_interaction_matrix, self.init_item_cf_embeds)
        return estimated_user_embeds
    
    def calculate_user_embedding(self, users):
        user_interaction_matrix = self.user_interact_matrix[users]
        user_interaction_matrix = self.csr_to_torch_sparse(user_interaction_matrix).to(self.device)
        estimated_user_embeds = torch.sparse.mm(user_interaction_matrix, self.init_item_cf_embeds)
        return estimated_user_embeds
    
    def cal_loss_origin(self, users, pos_items, neg_items):
        users_emb, all_items = self.compute_origin(users.cpu())
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        if(self.train_norm):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)

        numerator = torch.exp(pos_ratings / self.tau)

        denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim = 1)
        
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))
        return ssm_loss, users_emb, all_items
    
    def cal_loss_aug(self, users, pos_items, neg_items):
        users_emb, all_items = self.compute_aug(users.cpu())
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        if(self.train_norm):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        numerator = torch.exp(pos_ratings / self.tau)
        denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))
        return ssm_loss, users_emb, all_items
    
    def align_contrastive_loss(self, emb1, emb2, temperature=0.07):
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)

        logits = torch.matmul(emb1, emb2.T) / temperature
        labels = torch.arange(emb1.size(0)).long().to(emb1.device)

        loss_1 = F.cross_entropy(logits, labels) 
        loss_2 = F.cross_entropy(logits.T, labels)
        return (loss_1 + loss_2) / 2
    
    def forward(self, users, pos_items, neg_items):
        ssm_loss_1, users_emb1, all_items1 = self.cal_loss_origin(users, pos_items, neg_items)
        ssm_loss_2, users_emb2, all_items2 = self.cal_loss_aug(users, pos_items, neg_items)
        align_loss_u = self.align_contrastive_loss(users_emb1, users_emb2, self.align_temperature)
        align_loss_i = self.align_contrastive_loss(all_items1[pos_items], all_items2[pos_items], self.align_temperature)
        return ssm_loss_1 + ssm_loss_2 + self.align_loss * (align_loss_u + align_loss_i)

    @torch.no_grad()
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.data.n_items))

        users_emb, all_items = self.compute_origin(users)
        
        users = users_emb.to(self.device)
        items = all_items[torch.tensor(items).cuda(self.device)]
        
        if(self.pred_norm == True):
            users = F.normalize(users, dim = -1)
            items = F.normalize(items, dim = -1)
        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items) 

        return rate_batch.cpu().detach().numpy()