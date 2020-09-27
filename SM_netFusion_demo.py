#Code converted from MATLAB to Python by Birkan Ak, Istanbul Technical University
#Note that this is heavily influenced by NAGFS-PY
import warnings
warnings.filterwarnings("ignore") #We need to ignore some of the warnings because it gives warning in each iteration in sklearn validations.

import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from nxviz.plots import CircosPlot
import networkx as nx
from simulateData import simulate_data
from SM_netFusion import SM_netFusion
from scoreAcrossAllCVRuns import scoreAcrossAllCVRuns
#Variables below.

mu1 = 0.9
sigma1 = 0.49
mu2 = 0.7
sigma2 = 0.6
Nf = 5  # selected features
print("The number of selected features is automatically set to: ", Nf,
      "\nTo change it, please set Nf variable inside SM_netFusion_demo to a different integer.")
displayResults = 1
print(
    "The option for displaying the estimated atlases and selected features at each run of the cross-validation "
    "algorithm is set to:",
    displayResults)
if displayResults == 1:
    print("\nTo turn it off, please set displayResults variable inside SM_netFusion_demo to 0. \n")
print('Note that displaying the results at each run will slow down the demo. \n')

#We use user inputs here. This creates a matrix
data = simulate_data(mu1, sigma1, mu2, sigma2)

number = ()
number2 = ()
for i in range(len(data[2])):
    if data[2][i] == 1:
        number = number + (i,)
data_class1 = data[0][number, :]
# REFERENCE: NAGFS-PY
for i in range(len(data[2])):
    if data[2][i] == -1:
        number2 = number2 + (i,)
data_class2 = data[0][number2, :]
data_class11 = np.concatenate(data_class1)
data_class22 = np.concatenate(data_class2)
# * * (3) This part include  seperating samples by each classes.
#This part will used for plotting gaussian distribution of datas of each classes.
number=()
number2=()
#Data[0] (Featurematrix) and Data[2](LabelMatrix) has same order and  same index, so they can be seperated featurematrix
#with using labelmatrix.
for i in range(len(data[2])):
    if data[2][i]==1:
        number=number+(i,)
data_class1=data[0][number,:]

for i in range(len(data[2])):
    if data[2][i]==-1:
        number2=number2+(i,)
data_class2=data[0][number2,:]

#Gaussian Distribution needs 1x(NxK) matrixes, so they are converted to this shape.
data_class11=np.concatenate(data_class1)
data_class22=np.concatenate(data_class2)
# * *

k_fold = 5
predict_list = []
ind_array = np.empty((0, Nf), int)

print("\nPlease wait... This process takes some time according to user's inputs...")

#This is the main loop, most of the work done is here.
for i in range(0, (len(data[2]))):
    general_index = []
    print (i)
    for j in range(len(data[2])):
        general_index.append(j)
    #Here we divide training and testing samples from each other.
    general_index.remove(i)
    train_data = data[1][general_index, :, :]
    train_feature_data = data[0][general_index, :]
    train_Labels = data[2][general_index]
    test_data = data[1][i, :, :]
    test_feature_data = data[0][i, :]
    test_Label = data[2][i]

    #Use of SM_netFusion is below.
    AC1, AC2, ind = SM_netFusion(train_data, train_Labels, Nf, displayResults)

    #NOTE: I sikkiped the part with test_C1 and test_C2 since we don't use those arrays.


    #5 most discriminative features are indexed here.
    ind1=np.ravel(ind)
    ind_array=np.append(ind_array,[ind1.reshape(Nf)],axis=0)

    #Extract the top Nf discriminative training features

    delete_list = []
    cont2 = True
    cont4 = True
    for i in range(train_feature_data.shape[1]):
        for j in range(len(ind)):
            if i == ind[j]:
                cont2 = False
            else:
                continue
        if cont2 == True:
            delete_list.append(i)  # for delete unnecessary columns-features
        else:
            cont2 = True

    train_set = np.delete(train_feature_data, delete_list, 1)

    #Do the same steps for test!

    for i in range(len(test_feature_data)):
        for j in range(len(ind)):
            if i == ind[j]:
                cont4 = False
            else:
                continue
        if cont4 == True:
            delete_list.append(i)
        else:
            cont4 = True

    test_set = np.delete(test_feature_data, delete_list)
    test_set = test_set.reshape(-1, 1)
    test_set = test_set.transpose()

    #Use of SVM classifier. This is helpful for the predictions acquired from the test samples.
    clf = SVC(kernel="linear", C=1)
    clf.fit(train_set, train_Labels)
    pred = clf.predict(test_set)
    predict_list.append(pred)

#Accuracy, sensitivity and specificity is here.
conf=confusion_matrix(data[2],predict_list)
TN = conf[0][0]
FN = conf[1][0]
TP = conf[1][1]
FP = conf[0][1]
TPR = TP / (TP + FN) # Sensitivity
TNR = TN / (TN + FP) # Specificity
ACC = (TP+TN)/(TP+FP+FN+TN) # Accuracy
print("Confusion Matrix: ")
print(conf)
print("* * * * * ")
print("Accuracy Score: ",ACC)
print("Sensitivity Score: ",TPR)
print("Specificity Score: ",TNR)
# * *

# * * (10) Score_index function is used to find 5 most discriminative features across all iterations by scoring them.
Score_index=scoreAcrossAllCVRuns(ind_array)
# * *


# * * (11) In this part, 5 most discriminative features which are across all cross-validation runs are plotted in matrix.
aa11=data[1][0]
aa22=data[0][0]
last_coor=[]
for i in range(len(Score_index)):
    key1=True
    for j in range(aa11.shape[0]):
        for k in range(aa11.shape[1]):
            if aa22[Score_index[i]]==aa11[j][k]:
                if key1:
                    last_coor.append([j,k])
                    key1=False
                else:
                    continue
topScoreFeatures=np.zeros((aa11.shape[0],aa11.shape[1]))
s=0
ss=0
for i in range(len(Score_index)):
    topScoreFeatures[last_coor[i][0]][last_coor[i][1]]=  aa22[Score_index[s]]
    topScoreFeatures[last_coor[i][1]][last_coor[i][0]] = aa22[Score_index[s]]
    ss+=1
    if ss==2:
        s+=1
        ss=0


#Plotting feature matrix
plt.imshow(topScoreFeatures)
plt.title('Most Discriminative Features Across All Cross-Validation Runs')
plt.colorbar()
plt.show()
#* *
#* *  Plotting circular graph of top Nf discriminative features across all cross-validation runs
node_list=[]
edge_list=[]

for i in range(aa11.shape[0]):
    node=i+1
    node_list.append(node)
for i in range(len(last_coor)):

    last_coor[i][0] += 1
    last_coor[i][1] += 1

    edge_list.append(last_coor[i])
G = nx.Graph()
G.add_nodes_from(node_list)
G.add_edges_from(edge_list)
color_list=["a", "b", "c", "d", "e"]
for n, d in G.nodes(data=True):
    G.nodes[n]["class"] = node_list[n-1]

c = CircosPlot(graph=G,node_labels=True,
    node_label_rotation=True,
    fontsize=20,
    group_legend=True,
    figsize=(7, 7),node_color="class")

c.draw()
plt.title("circular graph of top Nf discriminative features across all cross-validation runs".title())
plt.show()


