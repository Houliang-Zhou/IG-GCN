# In[]
import numpy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix
import numpy as np
import torch
# print(torch.__version__)
from torch_geometric.data import Data
from tqdm import tqdm
import scipy.io as sio
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import read_csv
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def analysis_num_of_degrees(num_of_degrees):
    sns.displot(num_of_degrees)
    plt.show()

def separate_data_adnitype(dataset, disease_id, adnitype_id=0):
    '''

    Args:
        disease_id: 0: HC vs AD; 1: HC vs MCI; 2: MCI vs AD
        adnitype_id: 0

    Returns:

    '''
    test_data = []
    train_data = []
    for data in dataset:
        if data.adni_type == adnitype_id:
            if data.y > 0:
                data.y = torch.Tensor([1]).long()
            test_data.append(data)
        else:
            if disease_id==0:
                if data.y == 0 or data.y == 4:
                    if data.y > 0:
                        data.y = torch.Tensor([1]).long()
                    train_data.append(data)
            elif disease_id==1:
                if data.y == 0 or data.y == 1 or data.y == 2 or data.y == 3:
                    if data.y > 0:
                        data.y = torch.Tensor([1]).long()
                    train_data.append(data)
            elif disease_id==2:
                if data.y == 4 or data.y == 1 or data.y == 2 or data.y == 3:
                    if data.y >= 4:
                        data.y = torch.Tensor([1]).long()
                    else:
                        data.y = torch.Tensor([0]).long()
                    train_data.append(data)
    print('len of train: %d, and test:%d '%(len(train_data),len(test_data)))
    return train_data, test_data

def loadBrainImg_Snps_CSV(disease_id=0, path = './data/snps/data/%s/', k_inknn = 10):
    file_path = path
    if disease_id == 0:
        file_path_snps = file_path % ('data_AH')
    elif disease_id == 1:
        file_path_snps = file_path % ('data_MH')
    else:
        file_path_snps = file_path % ('data_AM')
    snps_data = read_csv(file_path_snps + 'snp.csv')
    snps_data = snps_data.values
    snps_data[np.isnan(snps_data)] = 0
    scaler = MinMaxScaler()#StandardScaler()
    scaled_snps_data = scaler.fit_transform(snps_data)

    file_path_img = file_path_snps
    file_path_img += "knn/%d/" % (k_inknn)
    BL_DXGrp_label = sio.loadmat(file_path_img + 'BL_DXGrp_label.mat')
    corr_data = sio.loadmat(file_path_img + 'corr_data.mat')
    imgData_mat = sio.loadmat(file_path_img + 'imgData_mat.mat')
    imgData_mat_normalized = sio.loadmat(file_path_img + 'imgData_mat_normalized.mat')
    BL_DXGrp_label = BL_DXGrp_label['BL_DXGrp_label']
    corr_data = corr_data['corr_data']
    imgData_mat = imgData_mat['imgData_mat']
    imgData_mat_normalized = imgData_mat_normalized['imgData_mat_normalized']
    imgData_mat[np.isnan(imgData_mat)] = 0
    imgData_mat_normalized[np.isnan(imgData_mat_normalized)] = 0

    d1, d2, d3 = imgData_mat_normalized.shape[0], imgData_mat_normalized.shape[1], imgData_mat_normalized.shape[2]
    imgData_mat = imgData_mat.reshape((d1, d2, -1))
    imgData_mat_normalized = imgData_mat_normalized.reshape((d1, d2, -1))
    normal_sum = np.sum(BL_DXGrp_label == 0)
    abnor_sum = np.sum(BL_DXGrp_label == 1)
    print('number of normal and abnormal:', normal_sum, abnor_sum)

    # img_scaler = MinMaxScaler()
    # imgData_mat = np.transpose(imgData_mat, (2,0,1))
    # imgData_mat_scaled = []
    # for i in range(d3):
    #     imgData_mat_scaled.append(img_scaler.fit_transform(imgData_mat[i]))
    # imgData_mat_scaled = np.asarray(imgData_mat_scaled)
    # imgData_mat_normalized = np.transpose(imgData_mat_scaled, (1,2,0))

    dataset = []
    for i in range(BL_DXGrp_label.shape[0]):
        snps_feat = torch.from_numpy(scaled_snps_data[i,:].reshape((1,-1))).float()
        X = torch.from_numpy(imgData_mat_normalized[i]).float()
        A = torch.from_numpy(corr_data[i]).float()
        y = torch.Tensor(BL_DXGrp_label[i, :]).long()
        A_coo = coo_matrix(A)
        edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
        edge_weight = torch.from_numpy(A_coo.data).float()
        data_one = Data(x=X, edge_index=edge_index, edge_attr=edge_weight, y=y,
                        A=A,
                        snps_feat=snps_feat)
        dataset.append(data_one)
    print('number of all data:', len(dataset))
    return dataset


def loadBrainImg_Snps_ADNI874(disease_id=0, path = './data/snps/data/preprocessing/', k_inknn = 5, num_cluster=2, isUseTsne4Similar = False, clinical_scores = -1, isPermutTest = False, Seed4PermutTest=1000,
                              isMultiFusion = False, isMultilModal4Similarity = False):
    file_path_img = path
    file_path_img += "knn/%d/" % (k_inknn)
    file_path_img_MRI = path + "MRI/knn/%d/" % (k_inknn)
    file_path_img_FDGPET = path + "FDG-PET/knn/%d/" % (k_inknn)
    file_path_img_AV45PET = path + "AV45-PET/knn/%d/" % (k_inknn)

    sbjID = sio.loadmat(file_path_img + 'sbjID_int.mat')
    sbjID = sbjID['sbjID_int']

    # 9 dimensions: label, age, edu, sex, abeta, tau, ptau, adas13, mmse
    scores_val_fill = sio.loadmat(file_path_img + 'score_excel_val_adni874_fill.mat')
    scores_val_fill = scores_val_fill['info_score_val_fill']
    scaler4score = MinMaxScaler()  # StandardScaler()
    scores_val_fill = scaler4score.fit_transform(scores_val_fill)
    if clinical_scores==-1:
        select_index = np.array([5, 7, 8])
        scores_regression = scores_val_fill[:, select_index]
        print("Clinical Scores Min:",np.min(scores_regression,0))
        print("Clinical Scores Max:",np.max(scores_regression,0))
    else:
        scores_regression = scores_val_fill[:, clinical_scores]
        scores_regression = np.expand_dims(scores_regression, 1)

    print("check infs or nan:",np.isnan(scores_val_fill).any())
    print("check infs or nan:",np.isinf(scores_val_fill).any())
    print("check infs or nan:",np.sum(np.isnan(scores_val_fill)))

    scores_val_missing = sio.loadmat(file_path_img + 'score_excel_val_adni874.mat')
    scores_val_missing = scores_val_missing['info_score_val_subs']

    clusters_pred_label = np.load(file_path_img+"center_%d/clusters_pred_label.npy"%(num_cluster)).astype(int)
    clusters_pred_label = np.expand_dims(clusters_pred_label,1)

    if isUseTsne4Similar:
        tsne_results_path = file_path_img + 'tsne_results.npy'
        tsne_results = np.load(tsne_results_path)
    else:
        if isMultilModal4Similarity:
            multimodal_for_similarity_path = file_path_img + 'multimodal_for_similarity.npy'
            tsne_results = np.load(multimodal_for_similarity_path)
        else:
            pet_for_similarity_path = file_path_img + 'pet_for_similarity.npy'
            tsne_results = np.load(pet_for_similarity_path)

    snps_data = sio.loadmat(file_path_img + 'SNPS_data.mat')
    snps_data = snps_data['SNPS_data']
    snps_data[np.isnan(snps_data)] = 0

    scaler = MinMaxScaler()#StandardScaler()
    scaled_snps_data = scaler.fit_transform(snps_data)
    # draw_snps(snps_data)

    BL_DXGrp_label = sio.loadmat(file_path_img + 'BL_DXGrp_label.mat')
    corr_data = sio.loadmat(file_path_img + 'corr_data.mat')
    imgData_mat = sio.loadmat(file_path_img + 'imgData_mat.mat')
    imgData_mat_normalized = sio.loadmat(file_path_img + 'imgData_mat_normalized.mat')
    BL_DXGrp_label = BL_DXGrp_label['BL_DXGrp_label']
    corr_data = corr_data['corr_data']
    imgData_mat = imgData_mat['imgData_mat']
    imgData_mat_normalized = imgData_mat_normalized['imgData_mat_normalized']
    imgData_mat[np.isnan(imgData_mat)] = 0
    imgData_mat_normalized[np.isnan(imgData_mat_normalized)] = 0

    BL_DXGrp_label -= 1

    d1, d2 = imgData_mat_normalized.shape[0], imgData_mat_normalized.shape[1]
    imgData_mat = imgData_mat.reshape((d1, d2, -1))
    imgData_mat_normalized = imgData_mat_normalized.reshape((d1, d2, -1))

    '''
    -------------------------------------------------------------------------
    '''
    '''
    info_score_subs label: HC=0, MCI=1, AD=2
    -------------------------------------------------------------------------
    '''
    if disease_id == 0:
        select_indices = np.where((BL_DXGrp_label == 0) | (BL_DXGrp_label == 4))[0]
    elif disease_id == 1:
        select_indices = \
            np.where((BL_DXGrp_label == 0) | (BL_DXGrp_label == 1) | (BL_DXGrp_label == 2) | (BL_DXGrp_label == 3))[0]
    elif disease_id == 2:
        select_indices = \
            np.where((BL_DXGrp_label == 4) | (BL_DXGrp_label == 2) | (BL_DXGrp_label == 3) | (BL_DXGrp_label == 1))[0]
    elif disease_id == 3:
        select_indices = \
            np.where((BL_DXGrp_label == 0) | (BL_DXGrp_label == 1) | (BL_DXGrp_label == 2) | (BL_DXGrp_label == 3) | (BL_DXGrp_label == 4))[0]
    '''
    -------------------------------------------------------------------------
    '''
    if isPermutTest:
        np.random.seed(Seed4PermutTest)
        scores_regression = np.random.permutation(scores_regression)
        scores_val_missing = np.random.permutation(scores_val_missing)

    BL_DXGrp_label = BL_DXGrp_label[select_indices]
    clusters_pred_label = clusters_pred_label[select_indices]
    tsne_results = tsne_results[select_indices]
    scores_regression = scores_regression[select_indices]
    scores_val_missing = scores_val_missing[select_indices]
    imgData_mat = imgData_mat[select_indices]
    corr_data = corr_data[select_indices]
    imgData_mat_normalized = imgData_mat_normalized[select_indices]
    scaled_snps_data = scaled_snps_data[select_indices]
    sbjID = sbjID[select_indices]

    if isMultiFusion:
        corr_data_mri = sio.loadmat(file_path_img_MRI + 'corr_data.mat')
        corr_data_mri = corr_data_mri['corr_data']
        corr_data_fdgpet = sio.loadmat(file_path_img_FDGPET + 'corr_data.mat')
        corr_data_fdgpet = corr_data_fdgpet['corr_data']
        corr_data_av45pet = sio.loadmat(file_path_img_AV45PET + 'corr_data.mat')
        corr_data_av45pet = corr_data_av45pet['corr_data']
        corr_data_mri = corr_data_mri[select_indices]
        corr_data_fdgpet = corr_data_fdgpet[select_indices]
        corr_data_av45pet = corr_data_av45pet[select_indices]

    if disease_id == 0:
        BL_DXGrp_label[BL_DXGrp_label > 0] = 1
    elif disease_id == 1:
        BL_DXGrp_label[BL_DXGrp_label > 0] = 1
    elif disease_id == 2:
        BL_DXGrp_label[BL_DXGrp_label == 1] = 0
        BL_DXGrp_label[BL_DXGrp_label == 2] = 0
        BL_DXGrp_label[BL_DXGrp_label == 3] = 0
        BL_DXGrp_label[BL_DXGrp_label == 4] = 1
    elif disease_id == 3:
        BL_DXGrp_label[BL_DXGrp_label == 1] = 1
        BL_DXGrp_label[BL_DXGrp_label == 2] = 1
        BL_DXGrp_label[BL_DXGrp_label == 3] = 1
        BL_DXGrp_label[BL_DXGrp_label == 4] = 2

    if disease_id < 3:
        normal_sum = np.sum(BL_DXGrp_label == 0)
        abnor_sum = np.sum(BL_DXGrp_label == 1)
        print('number of normal and abnormal:', normal_sum, abnor_sum)
    else:
        print('number of class 0, 1, and 2:', np.sum(BL_DXGrp_label == 0), np.sum(BL_DXGrp_label == 1), np.sum(BL_DXGrp_label == 2))

    print('number of cluster 0 and 1:', np.sum(clusters_pred_label == 0), np.sum(clusters_pred_label == 1))

    dataset = []
    for i in range(BL_DXGrp_label.shape[0]):
        sbjID_each = torch.Tensor(sbjID[i]).long()
        snps_feat = torch.from_numpy(np.expand_dims(scaled_snps_data[i,:], axis=0)).float()
        X = torch.from_numpy(imgData_mat_normalized[i]).float()
        A = torch.from_numpy(corr_data[i]).float()
        y = torch.Tensor(BL_DXGrp_label[i, :]).long()
        clust_y = torch.Tensor(clusters_pred_label[i, :]).long()
        clini_score = torch.Tensor(scores_regression[i, :]).float()
        demographics = torch.Tensor(scores_val_missing[i, :]).float()
        tsne_fdim = torch.from_numpy(np.expand_dims(tsne_results[i, :], axis=0)).float()
        A_coo = coo_matrix(A)
        edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
        edge_weight = torch.from_numpy(A_coo.data).float()
        if isMultiFusion:
            X = X.reshape((-1,1))
            Adj = torch.from_numpy(corr_data_mri[i]).float()
            edge_index_mri, edge_weight_mri = sparse_matrix(Adj)
            Adj = torch.from_numpy(corr_data_fdgpet[i]).float()
            edge_index_fdgpet, edge_weight_fdgpet = sparse_matrix(Adj)
            edge_index_fdgpet += d2
            Adj = torch.from_numpy(corr_data_av45pet[i]).float()
            edge_index_av45pet, edge_weight_av45pet = sparse_matrix(Adj)
            edge_index_av45pet += d2*2
            edge_index = torch.cat((edge_index_mri, edge_index_fdgpet, edge_index_av45pet),-1)
            edge_weight = torch.cat((edge_weight_mri, edge_weight_fdgpet, edge_weight_av45pet),-1)
        data_one = Data(x=X, edge_index=edge_index, edge_attr=edge_weight, y=y, clust_y=clust_y,
                        A=A,
                        snps_feat=snps_feat,
                        sbjID=sbjID_each,
                        tsne_fdim=tsne_fdim,
                        clini_score=clini_score,
                        demographics=demographics)
        dataset.append(data_one)
    print('number of all data:', len(dataset))
    print('isMultiFusion:',isMultiFusion, '; dim of feature matrix:', X.shape)
    return dataset, scaler4score

def draw_snps(snps):
    # snps = np.random.rand(100, 54)
    snps = snps[:50,:50]
    ax = sns.heatmap(snps, cmap='jet') #, linewidth=0.1
    plt.show()

def sparse_matrix(Adj):
    A_coo = coo_matrix(Adj)
    edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
    edge_weight = torch.from_numpy(A_coo.data).float()
    return edge_index, edge_weight


class ADNIDataset(InMemoryDataset):
    def __init__(self, root, name, dataset, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False):
        self.name = name
        self.cleaned = cleaned
        super(ADNIDataset, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = self.process_trans(dataset)
        #self.num_features = 7
        # if self.data.x is not None and not use_node_attr:
        #     num_node_attributes = 0
        #     self.data.x = self.data.x[:, num_node_attributes:]
        # if self.data.edge_attr is not None and not use_edge_attr:
        #     num_edge_attributes = 0
        #     self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def num_features(self):
        return 3

    @property
    def num_classes(self):
        return 2

    def process_trans(self, dataset):
        new_data_list = []
        for data in tqdm(dataset):
            new_data_list.append(self.pre_transform(data))
        data_list = new_data_list
        self.data, self.slices = self.collate(data_list)
        return self.data, self.slices

if __name__ == "__main__":
    # loadBrainImg(disease_id=3, isShareAdj=False, isInfo_Score=True, isSeperatedGender=False, selected_gender=1)
    # loadBrainImg_Snps_CSV(disease_id=0, path = './data/snps/data/%s/', k_inknn = 10)
    loadBrainImg_Snps_ADNI874(disease_id=3, path = './data/snps/data/preprocessing/', k_inknn = 5, isUseTsne4Similar=False, isMultiFusion=True)