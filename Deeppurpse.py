import os , json, datetime, argparse
from DeepPurpose import utils,models,dataset,property_pred
from rdkit.Chem import PandasTools,Draw
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from ax import ParameterType, ChoiceParameter, RangeParameter,  SearchSpace, Experiment, modelbridge
from plotly.offline import plot
from ax.plot.contour import interact_contour,plot_contour
from ax.plot.slice import plot_slice
import matplotlib.pyplot as plt
from ax.plot.scatter import interact_fitted
import time

# 读取指定的文件，返回两个列表，positive正例、negative反例
def get_smilse(file):
    positive = []
    negative = []
    file = open(file)
    for line in file.readlines():
        lineArr = line.strip().split()
        if int(lineArr[1]) == 1:
            positive.append(lineArr[0])
        else:
            negative.append(lineArr[0])
    return positive, negative

# 从文件中读取所有smiles字符串，并将他们保存在一个列表中
def get_all_smilse(file):
    all_smiles = []
    file = open(file)
    for line in file.readlines():
        lineArr = line.strip().split()
        all_smiles.append(lineArr[0])
    return all_smiles

# 该函数将SMILES字符串转换为RDKit的分子对象（Mol对象）
def get_mols(smiles):
    mols = [Chem.MolFromSmiles(mol) for mol in smiles]
    return  mols

# 这个函数将SDF文件（结构数据文件）转换为SMILES格式，并将转换结果保存为SMILES文件
def converter(file_name):
    mols = [mol for mol in Chem.SDMolSupplier(file_name)]
    outname = file_name.split(".sdf")[0] + ".smi"
    out_file = open(outname, "w")
    for mol in mols:
        smi = Chem.MolToSmiles(mol)
        name = mol.GetProp("_Name")
        out_file.write("{}\t{}\n".format(smi, name))
    out_file.close()



positive, negative = get_smilse('./total数据.txt')

pos_mols = get_mols(positive)
DRAW_MOLS_TO_GRID_IMAGE = Draw.MolsToGridImage(pos_mols[:40], molsPerRow=5, subImgSize=(300, 300))
neg_mols = get_mols(negative)
mols = pos_mols + neg_mols


##
flag =0
# flag = 1
if flag:
    img_pos = DRAW_MOLS_TO_GRID_IMAGE
    img_pos.save("molecules_positive.png")
    img_neg = Draw.MolsToGridImage(neg_mols[:40], molsPerRow=5,subImgSize=(300, 300))
    img_neg.save("molecules_negative.png")


##
flag=0
flag=1
if flag:
    morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]
    df_morgan_fps = pd.DataFrame(np.array(morgan_fps))
    kmeans = KMeans(n_clusters=8,n_init='auto')
    kmeans.fit(df_morgan_fps)
    pca = PCA(n_components=2)
    decomp = pca.fit_transform(df_morgan_fps)
    x = decomp[:, 0]
    y = decomp[:, 1]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, c=kmeans.labels_,s=10,alpha=0.7)
    plt.title("PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar()
    plt.show()


flag = 0
if flag:
    result_file = open('vitrual.out', 'w')
    vitrual_drug = get_all_smilse(r'D:\Drug\new\Date_test_13.txt')
    number = len(vitrual_drug)
    drug_encoding = 'MPNN'
    y = np.ones(number)
    model = property_pred.model_pretrained(r'D:\Drug\new\model_GPU_SEED')
    x_pred = utils.data_process(X_drug=vitrual_drug, y=y, drug_encoding=drug_encoding, split_method='no_split')
    start_time = time.time()
    y_pred = model.predict(x_pred)
    end_time = time.time()
    run_time = end_time - start_time
    print('time cost:   ' + str(run_time))
    vitrual_drug_d = {}
    for i in range(len(x_pred)):
        name = str(x_pred['SMILES'][i])
        key = y_pred[i]
        vitrual_drug_d[name] = float(key)
    sort_vitrual = sorted(vitrual_drug_d.items(), key=lambda e: e[1], reverse=True)
    batch_size = 50000
    output_file = 'output.txt'
    result_file = open(output_file, 'a')
    for i, item in enumerate(sort_vitrual):
        result_file.write(str(item) + '\n')
        if (i + 1) % batch_size == 0:
            result_file.flush()
            print(f'Processed {i + 1} items')
    result_file.close()



    # 将格式化后的结果写入新文件
    formatted_output_file = 'formatted_output.txt'
    with open(formatted_output_file, 'w') as file:
        for line in formatted_lines:
            file.write(line + '\n')
##
flag=0
if flag:
    X_drugs,X_targets,y =dataset.read_file_training_dataset_bioassay('./数据1.txt')
    drug_encoding='MPNN'
    train,val,test=utils.data_process(X_drug=X_drugs,y=y,drug_encoding=drug_encoding,split_method='random',frac=[0.8,0.1,0.1],random_seed=2)
    train.head(1)
    config = utils.generate_config(drug_encoding=drug_encoding,
                                   cls_hidden_dims=[256,512,256],train_epoch=50,LR=0.002,batch_size=15,
                                   hidden_dim_drug=32,mpnn_depth=4,mpnn_hidden_size=128)
    model= property_pred.model_initialize(**config)
    model.train(train,val,test)
    model.save_model('./model')


flag=0
if flag:
    exp = ['OC1=CC=C(C(C)(C)C2=CC=C(O)C=C2)C=C1']
    m = Chem.MolFromSmiles('OC1=CC=C(C(C)(C)C2=CC=C(O)C=C2)C=C1')
    draw = Draw.MolToImage(m)
    draw.save('exp.png')
    drug_encoding = 'MPNN'
    model = property_pred.model_pretrained('./model_GPU_real_3')
    y=[1]
    x_pred = utils.data_process(X_drug=exp,y=y, drug_encoding=drug_encoding,split_method='no_split')
    y_pred=model.predict(x_pred)
    print(y_pred)



flag=0
if flag:
    result_file = open('vitrual.out','w')
    vitrual_drug = get_all_smilse('./Date_test_13.txt')
    number = len(vitrual_drug)
    drug_encoding = 'MPNN'
    y = np.ones(number)
    model = property_pred.model_pretrained(r'D:\Drug\new\model_GPU_SEED')
    x_pred = utils.data_process(X_drug=vitrual_drug, y=y, drug_encoding=drug_encoding, split_method='no_split')
    start_time = time.time()
    y_pred = model.predict(x_pred)
    end_time = time.time()
    run_time = end_time - start_time
    print('time cost:   '+str(run_time))
    vitrual_drug_d = {}
    for i in range(len(x_pred)):
        name = str(x_pred['SMILES'][i])
        key = y_pred[i]
        vitrual_drug_d[name] = float(key)
    sort_vitrual = sorted(vitrual_drug_d.items(), key = lambda e:e[1], reverse=True)
    for item in sort_vitrual:
        result_file.write(str(item)+'\n')

flag=0
if flag:
    exp = ['FC1=CC([C@H]2OC([H])[C@H](N3CC4=CN(S(=O)(C)=O)N=C4C3)[C@H](F)[C@@H]2N)=C(F)C=C1[H]']
    drug_encoding = 'MPNN'
    model = property_pred.model_pretrained(r'D:\Drug\new\model_GPU_SEED')
    print(model.config)
    y = [5]
    x_pred = utils.data_process(X_drug=exp,y=y, drug_encoding=drug_encoding,split_method='no_split')
    y_pred = model.predict(x_pred)
    print(y_pred)

flag = 0
if flag:
    result_file = open('vitrual.out', 'w')
    vitrual_drug = get_all_smilse('D:\7z\new\D001-1_1.smi')
    number = len(vitrual_drug)
    drug_encoding = 'MPNN'
    y = np.ones(number)
    model = property_pred.model_pretrained(r'D:\7z\new\model_GPU_SEED')
    x_pred = utils.data_process(X_drug=vitrual_drug, y=y, drug_encoding=drug_encoding, split_method='no_split')
    start_time = time.time()
    y_pred = model.predict(x_pred)
    end_time = time.time()
    run_time = end_time - start_time
    print('time cost:   ' + str(run_time))
    vitrual_drug_d = {}
    for i in range(len(x_pred)):
        name = str(x_pred['SMILES'][i])
        key = y_pred[i]
        vitrual_drug_d[name] = float(key)
    sort_vitrual = sorted(vitrual_drug_d.items(), key=lambda e: e[1], reverse=True)
    batch_size = 50000
    output_file = 'output.txt'
    result_file = open(output_file, 'a')
    for i, item in enumerate(sort_vitrual):
        result_file.write(str(item) + '\n')
        if (i + 1) % batch_size == 0:
            result_file.flush()
            print(f'Processed {i + 1} items')
    result_file.close()

    # 过滤并格式化每一行
    formatted_lines = []
    for line in sort_vitrual:
        item = line[1]
        if item > 0.5:
            formatted_lines.append(line[0])

    # 将格式化后的结果写入新文件
    formatted_output_file = 'formatted_output.txt'
    with open(formatted_output_file, 'w') as file:
        for line in formatted_lines:
            file.write(line + '\n')

flag = 0
if flag:
    result_file = open('vitrual.out', 'w')
    vitrual_drug = get_all_smilse('D001-4.smi')
    number = len(vitrual_drug)
    drug_encoding = 'MPNN'
    y = np.ones(number)
    model = property_pred.model_pretrained(r'D:\7z\new\model _GPU_real_5')
    x_pred = utils.data_process(X_drug=vitrual_drug, y=y, drug_encoding=drug_encoding, split_method='no_split')
    start_time = time.time()
    y_pred = model.predict(x_pred)
    end_time = time.time()
    run_time = end_time - start_time
    print('time cost:   ' + str(run_time))
    vitrual_drug_d = {}
    for i in range(len(x_pred)):
        name = str(x_pred['SMILES'][i])
        key = y_pred[i]
        vitrual_drug_d[name] = float(key)
    sort_vitrual = sorted(vitrual_drug_d.items(), key=lambda e: e[1], reverse=True)
    batch_size = 50000
    output_file = 'output.txt'
    result_file = open(output_file, 'a')
    for i, item in enumerate(sort_vitrual):
        result_file.write(str(item) + '\n')
        if (i + 1) % batch_size == 0:
            result_file.flush()
            print(f'Processed {i + 1} items')
    result_file.close()

    # 过滤并格式化每一行
    formatted_lines = []
    for line in sort_vitrual:
        item = line[1]
        if item > 0.5:
            formatted_lines.append(line[0])

    # 将格式化后的结果写入新文件
    formatted_output_file = 'formatted_output.txt'
    with open(formatted_output_file, 'w') as file:
        for line in formatted_lines:
            file.write(line + '\n')