
"""
Name : Sourav Yadav 
ID : A20450418
CS584-04 Spring 2020
Assignment 3

"""


import numpy 
import pandas
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
from itertools import combinations
import math
global depth

global node_split
global rflag
global rdepth 
global ldepth

global depth
global rt
rt=[]
rflag=False
depth=0
rdepth=0
ldepth=0
node_split=[]


claims = pandas.read_csv('C:\\Users\\soura\\OneDrive\\Desktop\\ML\\ML Spring20_Assignemnt\\HW3\\claim_history.csv',
                            delimiter=',')

claim_data = claims[['CAR_TYPE', 'OCCUPATION', 'EDUCATION']]
claim_target=claims[['CAR_USE']]
print("\n--------------------------------Question1--------------------------------")
#X_train, X_test, y_train, y_test = train_test_split(claim_data, claim_target, test_size=0.25, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(claim_data, claim_target, test_size = 0.25, random_state = 60616, stratify = claim_target)

print("\n-------------------------Part A--------------------------")
print("Frequency Table of Target Varibale in Train")
print("Counts")
print(Y_train['CAR_USE'].value_counts())
print("Proportion")
print(Y_train['CAR_USE'].value_counts(normalize=True))


print("\n-------------------------Part B--------------------------")
print("Frequency Table of Target Varibale in Test")
print("Counts")
print(Y_test['CAR_USE'].value_counts())
print("Proportion")
print(Y_test['CAR_USE'].value_counts(normalize=True))

print("\n-------------------------Part C--------------------------")
#print(Y_train['CAR_USE'].value_counts()[1])
#print(claim_target['CAR_USE'].value_counts()[1])
print("The probability that an observation is in the Training partition given that CAR_USE = Commercial ::")
print(Y_train['CAR_USE'].value_counts()[1]/claim_target['CAR_USE'].value_counts()[1])
#print(Y_train['CAR_USE'].value_counts())
#print(claim_target['CAR_USE'].value_counts())

print("\n-------------------------Part D--------------------------")
print("The probability that an observation is in the Training partition given that CAR_USE = Private ::")
print(Y_test['CAR_USE'].value_counts()[0]/claim_target['CAR_USE'].value_counts()[0])

print("\n---------------------------------Question2--------------------------------")

#Calculate Entropy
def calEntropy(p1):
    e=0
    for i in p1:
        if i==0:
            continue            
        e+=(-i*math.log2(i))        
    return e


result_XY = pandas.concat([X_train, Y_train], axis=1, sort=False)

#Calculate Probabilty 
def calProba(x,y):
    normf=x+y
    return (x/normf,y/normf)


result=result_XY[result_XY['OCCUPATION'].isin(['Doctor', 'Clerical', 'Manager', 'Professional', 'Lawyer', 'Home Maker'])]
#print(result["CAR_USE"].value_counts())
left=result[result['CAR_TYPE'].isin(['Pickup', 'Panel Truck', 'Van'])]
#print(left["CAR_USE"].value_counts())
right=result[result['CAR_TYPE'].isin(['SUV', 'Minivan', 'Sports Car'])]
#print(right["CAR_USE"].value_counts())

#Decision Tree
def GenerateTree(result_XY):
    global depth 
    global ldepth
    global rdepth
    global rflag
    global node_split
    global rt
    if depth>=2:
        if rflag==False:
            depth=1
            rflag=True
        return

        
    temp=[]
    highest=0


    xt3=result_XY['CAR_USE'].value_counts()
    xt3.sort_index(inplace=True)
    tt3=pandas.Series([0,0],index=['Commercial', 'Private'])  
    for i in xt3.index:
        tt3[i]=xt3[i]  

    priv_p=tt3['Private']/tt3.sum()
    com_p=tt3['Commercial']/tt3.sum()
    parent_entropy=calEntropy([priv_p,com_p])
    
    re1,best_split1=selectOrdinSplit(result_XY,'EDUCATION',parent_entropy)
    temp.append((re1,best_split1,'EDUCATION'))

    re2,best_split2=selectNominSplit(result_XY,'OCCUPATION',parent_entropy)
    temp.append((re2,best_split2,'OCCUPATION'))
            
    re3,best_split3=selectNominSplit(result_XY,'CAR_TYPE',parent_entropy)
    temp.append((re3,best_split3,'CAR_TYPE'))
    #print(temp)       
    for i in range(3):        
        if temp[i][0]>highest:
            highest=temp[i][0]
            bs=temp[i][1]
            col_name=temp[i][2]
        else:
            continue
    node_split.append(bs)
    print(f"-----------Node at Depth {depth} -------------------")
    print(f"Split Criterion:")
    print(f"Predicator Name : {col_name}")
    print("----------Left Node------------")
    print("Left Node ",bs[0])
    left=result_XY[result_XY[col_name].isin(bs[0])]
    t1=left["CAR_USE"].value_counts()
    t1.sort_index(inplace=True)
    print("Left Node Traget Values:")
    print(t1)
    c,p=calProba(t1[0],t1[1])
    print("Left Node Probabilities:")
    print(f"Commercial:{c}\n Private: {p}")
    print("----------Right Node------------")
    print("Right Node",bs[1])
    right=result_XY[result_XY[col_name].isin(bs[1])]
    if rflag==False :
        rt.append(right)
#        print(rt)
    
        
    t2=right["CAR_USE"].value_counts()
    t2.sort_index(inplace=True)
    print("Right Node Traget Values:")
    print(t2)
    c1,p1=calProba(t2[0],t2[1])
    print("Right Node Probabilities:")
    print(f"Commercial:{c1}\n Private: {p1}")
    print(f"Entropy: {parent_entropy}")
    print(f"Split Entropy :{parent_entropy-highest}")
    #print(f"Private Values: {tt3['Private']}")
    #print(f"Commercial Values: {tt3['Commercial']}")
    if tt3['Commercial']>tt3['Private']:
        print("Predicted Class is Commercial")
    else:
        print("Predicted Class is Private")

    depth+=1
    GenerateTree(left)    
    GenerateTree(rt[0])

    


#Function for nominal Split
def selectNominSplit(result_XY,col_name,parent_entropy):
    cart=result_XY[col_name].unique()
#    cart=predictor_column.unique()
    cart_temp=cart.tolist()
    best_split=[0,0]
    minimum=0
    for i in range(int((len(cart)-1))):
        for j in combinations(cart,i+1): 
            split1=list(j)
            split2=list(set(cart_temp)-set(split1))
        
            t2=result_XY[[col_name,'CAR_USE']][result_XY[col_name].isin(split1)]
            t3=result_XY[[col_name,'CAR_USE']][result_XY[col_name].isin(split2)]
            #print(t2)
            xt1=t2['CAR_USE'].value_counts()
            xt2=t3['CAR_USE'].value_counts()
            xt1.sort_index(inplace=True)
            xt2.sort_index(inplace=True)
            tt1=pandas.Series([0,0],index=['Commercial', 'Private'])
            tt2=pandas.Series([0,0],index=['Commercial', 'Private'])
            for i in range(len(xt1.index)):
#            for i in xt1.index:
                tt1[i]=xt1[i]
           
            for j in range(len(xt2.index)):
#            for i in xt2.index:
                tt2[j]=xt2[j]
                
            split2_p1=tt2['Commercial']/tt2.sum()
            split2_p2=tt2['Private']/tt2.sum()
            split1_p1=tt1['Commercial']/tt1.sum()
            split1_p2=tt1['Private']/tt1.sum()  

            split1_e=calEntropy([split1_p1,split1_p2])
            split2_e=calEntropy([split2_p1,split2_p2])
            norm=len(t2.index)+len(t3.index)
            #split_entropy=(len(t2.index)/len(result_XY.index))*split1_e+(len(t3.index)/len(result_XY.index))*split2_e
            split_entropy=((len(t2.index)/norm)*split1_e)+((len(t3.index)/norm)*split2_e)
            reduced_entropy=parent_entropy-split_entropy
            if reduced_entropy>minimum:
                minimum=reduced_entropy
                best_split[0]=split1
                best_split[1]=split2
            #print("REntropy",minimum)
            #print(best_split)
            #print(reduced_entropy)
                
    return minimum, best_split

#Function of sorting in the order ['Below High School','High School', 'Bachelors', 'Masters', 'Doctors']
def sortOrder(in_list):
    new=['Below High School','High School', 'Bachelors', 'Masters', 'Doctors']
    cart_temp=[]
    for j in new:
        for i in in_list:
            if j==i:
                cart_temp.append(j)
    return cart_temp

#Function for Ordinal Split
def selectOrdinSplit(result_XY,col_name,parent_entropy):
    cart=result_XY[col_name].unique()
#    cart=predictor_column.unique()
    new=['Below High School','High School', 'Bachelors', 'Masters', 'Doctors']
    cart_temp=sortOrder(cart)
  
    best_split=[0,0]
    minimum=0
    for i in range(len(cart)-1):
#        for j in combinations(cart,i+1): 
            split1=cart_temp[:i+1]
            split2=list(set(cart_temp)-set(split1))
            split2=sortOrder(split2)
       
            t2=result_XY[[col_name,'CAR_USE']][result_XY[col_name].isin(split1)]
            t3=result_XY[[col_name,'CAR_USE']][result_XY[col_name].isin(split2)]
            xt1=t2['CAR_USE'].value_counts()
            xt2=t3['CAR_USE'].value_counts()
            xt1.sort_index(inplace=True)
            xt2.sort_index(inplace=True)
            tt1=pandas.Series([0,0],index=['Commercial', 'Private'])
            tt2=pandas.Series([0,0],index=['Commercial', 'Private'])
            for i in range(len(xt1.index)):
                tt1[i]=xt1[i]
           
            for j in range(len(xt2.index)):
                tt2[j]=xt2[j]
                
            split2_p1=tt2['Commercial']/tt2.sum()
            split2_p2=tt2['Private']/tt2.sum()
            split1_p1=tt1['Commercial']/tt1.sum()
            split1_p2=tt1['Private']/tt1.sum()  

            split1_e=calEntropy([split1_p1,split1_p2])
            split2_e=calEntropy([split2_p1,split2_p2])
            norm=len(t2.index)+len(t3.index)
            split_entropy=((len(t2.index)/norm)*split1_e)+((len(t3.index)/norm)*split2_e)
            #split_entropy=(len(t2.index)/len(result_XY.index))*split1_e+(len(t3.index)/len(result_XY.index))*split2_e
            reduced_entropy=parent_entropy-split_entropy
            if reduced_entropy>minimum:
                minimum=reduced_entropy
                best_split[0]=split1
                best_split[1]=split2

                
    return minimum, best_split

#Predict Class Probabability
def predict_class(predData):
    out_data = []
    if predData['OCCUPATION'] in ('Blue Collar', 'Student', 'Unknown'):
        if predData['EDUCATION'] <=0.5:              
            return [167/(167+453), 453/(167+453)]
        else:           
            return [1904/(1904+369), 369/(1904+369)]
    else:
        if predData['CAR_TYPE'] in ('Minivan', 'SUV', 'Sports Car'):

            return [29/(29+3415), 3415/(29+3415)]

        else:

            return [742/(742+647), 647/(742+647)]



def predict_class_decision_tree(predData):
    out_data = numpy.ndarray(shape=(len(predData), 2), dtype=float)
    counter = 0
    for index, row in predData.iterrows():
        probability = predict_class(predData=row)
        out_data[counter] = probability
        counter += 1
    return out_data


result_XY = pandas.concat([X_train, Y_train], axis=1, sort=False)
print(result_XY['CAR_USE'].value_counts())
priv_p=result_XY['CAR_USE'].value_counts()[0]/result_XY['CAR_USE'].value_counts().sum()
com_p=result_XY['CAR_USE'].value_counts()[1]/result_XY['CAR_USE'].value_counts().sum()
print("----------------------------Part a----------------------------\n")
print("Entropy of Root Node:")
parent_entropy=calEntropy([priv_p,com_p])
print(parent_entropy)

GenerateTree(result_XY)
#print(node_split)
print("----------------------------Part B----------------------------\n")
print("Calculated Above")
print("----------------------------Part C----------------------------\n")
print("Calculated Above")
print("----------------------------Part D----------------------------\n")
print("Calculated Above")
print("----------------------------Part E----------------------------\n")
print("Calculated Above")
print("--------------------------Part F------------------------------------")


Y_test=Y_test.to_numpy()

X_test['EDUCATION'] = X_test['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})
predProb_test = predict_class_decision_tree(predData=X_test)
predProb_y = predProb_test[:,0] 

# Generate the coordinates for the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(Y_test, predProb_y, pos_label = 'Commercial') 

max_tpr_fpr = numpy.where(tpr-fpr == max(tpr-fpr))[0][0]
cut_off = thresholds[[max_tpr_fpr]][0]
print(f"KS Cut Off Probabilty:{cut_off}")
print("Kolmogorov Smirnov value",max(tpr-fpr)) 
                       
# Draw the Kolmogorov Smirnov curve
cutoff = numpy.where(thresholds > 1.0, numpy.nan, thresholds)
plt.figure(figsize=(15,10))
plt.plot(cutoff, tpr, marker = 'o', label = 'True Positive',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(cutoff, fpr, marker = 'o', label = 'False Positive',
         color = 'red', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True, fontsize = 'large')
plt.show()  




print("---------------------------------Question3-----------------------------------------")

print("----------------------------------Part A-------------------------------------------")
#Taking Target Event Varibale as Commercial 
#Event (Commercial) probabilty is calculated as 0.367849 for Commercial in Question1 PartA
threshold=0.367849

# determining the predicted class
pred_y = numpy.empty_like(Y_test)
#pred_y = numpy.empty(Y_test.shape,dtype=str)
for i in range(Y_test.shape[0]):
    if predProb_y[i] > threshold:
        pred_y[i] = 'Commercial'
    else:
        pred_y[i] = 'Private'




# Calculating accuracy and misclassification rate
accuracy = metrics.accuracy_score(Y_test, pred_y)
misclassification_rate = 1 - accuracy
print("Train Proportion Misclassification Rate:")
print(f'Accuracy: {accuracy}')
print(f'Misclassification Rate: {misclassification_rate}')

print("------------------------------------Part B-------------------------------")

#
threshold1=max(tpr-fpr)

# Determining the predicted class
pred_y = numpy.empty_like(Y_test)
#pred_y = numpy.empty(Y_test.shape,dtype=str)
for i in range(Y_test.shape[0]):
    if predProb_y[i] > threshold1:
        pred_y[i] = 'Commercial'
    else:
        pred_y[i] = 'Private'

# Calculating accuracy and misclassification rate
accuracy = metrics.accuracy_score(Y_test, pred_y)
misclassification_rate = 1 - accuracy
print("KS Misclassification Rate:")
print(f'Accuracy: {accuracy}')
print(f'Misclassification Rate: {misclassification_rate}')
           
   
print("---------------------------Part C-----------------------------------")

# Calculating the root average squared error
RASE = 0.0
for y, ppy in zip(Y_test, predProb_y):
    if y == 'Commercial':
        RASE += (1 - ppy) ** 2
    else:
        RASE += (0 - ppy) ** 2
RASE = numpy.sqrt(RASE / Y_test.shape[0])
print(f'Root Average Squared Error: {RASE}')                         


print("-----------------------------Part D-------------------------------------")
y_true = 1.0 * numpy.isin(Y_test, ['Commercial'])
AUC = metrics.roc_auc_score(y_true, predProb_y)
print(f'Area Under Curve: {AUC}')
   
print("-----------------------------Part E-------------------------------------")
    
target_and_predictedprob=numpy.concatenate((Y_test, predProb_y.reshape(Y_test.shape)), axis=1)

com=target_and_predictedprob[target_and_predictedprob[:,0] == 'Commercial']
com[:,1]=numpy.sort(com[:,1])
priv=target_and_predictedprob[target_and_predictedprob[:,0] == 'Private']
priv[:,1]=numpy.sort(priv[:,1])


con=0
dis=0
tie=0

for i in com[:,1]:
    for j in priv[:,1]:
        if i>j:
            con+=1
        elif i==j:
            tie+=1
        else:
            dis+=1

            
print("Gini Coeffiecient:")  
pairs=con+dis+tie
print((con-dis)/pairs)

print("---------------------------Part F--------------------------")
print("Goodman-Kruskal Gamma statistic :")
print((con-dis)/(con+dis))


print("------------------------Part G----------------------------")
  
# Generate the coordinates for the ROC curve
one_minus_specificity, sensitivity, thresholds = metrics.roc_curve(Y_test, predProb_y, pos_label='Commercial')

# Add two dummy coordinates
one_minus_specificity = numpy.append([0], one_minus_specificity)
sensitivity = numpy.append([0], sensitivity)

one_minus_specificity = numpy.append(one_minus_specificity, [1])
sensitivity = numpy.append(sensitivity, [1])

# Draw the ROC curve
plt.figure(figsize=(6, 6))
plt.plot(one_minus_specificity, sensitivity, marker='o', color='blue', linestyle='solid', linewidth=2, markersize=6)
plt.plot([0, 1], [0, 1], color='red', linestyle=':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show() 
