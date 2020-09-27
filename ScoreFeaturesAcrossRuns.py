import numpy as  np

# * * (10.1) This function compute discrimination score of 5 discriminative features from all leave one out iteration steps.
#They are scored by their disrimination rankings  and  number of repetitions.
def Score_features(ind_array):

    Cf=[]
    Cf_sub=[]
    Score_list=[]
    Score_list1=[]
    Score_index=[]
    Score=0
    ind_array=ind_array.transpose()
    ind_array1=np.concatenate(ind_array)
    ind_array2=np.unique(ind_array1)
    for i in range(len(ind_array2)):
        for j in range(ind_array.shape[0]):
            for k in range(ind_array.shape[1]):
                if ind_array2[i]==ind_array[j][k]:
                    Cf_sub.append([j, k])
        Cf.append(Cf_sub)
        Cf_sub=[]
    for i in range(len(Cf)):
        for j in range(len(Cf[i])):

            Score+=5-Cf[i][j][0]
        Score_list.append(Score)
        Score_list1.append(Score)
        Score=0
    Score_list1.sort()
    Score_list1=Score_list1[::-1]
    con11=True
    for i in range(5):
        for j in range(len(Score_list)):
            if con11:
                if Score_list1[i]==Score_list[j]:
                    if not ind_array2[j] in Score_index:
                        Score_index.append(ind_array2[j])
                        con11=False
                    else:
                        j+=1
        con11=True
    Score_index=np.unique(Score_index)

#* *
    return Score_index

