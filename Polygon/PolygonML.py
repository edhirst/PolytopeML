'''Process Polygon data, select investigations to run, perform ML'''
'''Code comprised of 4 cells: 
    ~ Cell 1 --> Choose investigation parameters
    ~ Cell 2 --> Import libraries & data
    ~ Cell 3 --> Define the ML function
    ~ Cell 4 --> Iterate through selected investigations performing the ML
    To run: Select the investigations to consider in cell 1, run cell 1, run cell 2, run cell 3, run cell 4,
            ...then can repeatedly edit investigations in cell 1 then running cells 1 & 4 sequentially to perform different ML investigations.
    notes:  ensure path to datafile is correct on line 33, there is functionality for padding (although not used in investigations as performed worse),
    ...can input padding choice into function directly in cell 4, additional printing options at end of ML function and end of cell 4 which can be uncommented.
'''
#Select investigations to run
input_set = [1,2]       #...select 1 --> vertices, or 2 --> pluckers, or both
output_set = [7,8,9,10] #...select 7 --> volume, 8 --> dual volume, 9 --> gorenstein index, 10 --> codimension, or any combination thereof
number_vertex_set = [3,4,5,6] #...select which datasets of polygons with this many vertices to consider, note including '0' will run ML on all polygons using vector padding (can edit padding functionality in function call: what choice of number to pad with)
crossval_check = True   #...select whether to perform cross-validation, default is 5-fold (can edit in function directly)
tt_ratio_set = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]    #...if running over varying train/test split ratios (where no cross-validation, st k=1) select the train set proportions to consider from the interval (0,1)
gcd_scheme = 0          #...select which gcd augmentaion scheme to use: 0 --> none, 1 --> pairwise gcds, 2 --> (n-1)-gcds
inversion_check = False #...choose whether to run the inversion investigation, swapping the final vector entry with the learned parameter

#%%
#Import libraries
import numpy as np
from ast import literal_eval
from math import floor
from itertools import chain, combinations
from copy import deepcopy
import sqlite3
import pandas as pd
from tensorflow import keras

#Import Polygon Data
with sqlite3.connect("./dim_2_plucker.db") as db: #...ensure this is the correct path to the datafile
    c = db.cursor()
    df = pd.read_sql_query("SELECT * FROM dim_2_plucker", db)   #...read database into a pandas dataframe
    headings = df.columns.values                                #...save the headings of each column in the table
    data = df.values                                            #...convert pandas dataframe to np.array
del(c,df)
#Reformat dual volume data to floats
for polygon in data:
    if isinstance(polygon[8],str): #...where dual volume interpreted as string convert to a float
        polygon[8] = float(polygon[8].split('/')[0])/float(polygon[8].split('/')[1])
del(polygon)

#Extract the ranges of the polygon parameters
Y_Ranges = [[min([poly[i] for poly in data]),max([poly[i] for poly in data])] for i in [3,4,5,6,7,8,9,10]]


##############################################################################
#%% #ML data - set-up general function for all investigations
def ML(X=2,Y=7,poly_n=0,Pad=1,Pchoice=0,gcd=0,k_cv=5,split=0.8):
    #Extract relevant parts of data to ML 
    #Data selection hyper-params
    X_choice, Y_choice = X, Y   #...choose what to ML (use 'headings': id, vertices, plucker, plucker_len, num_vertices, num_points, num_interior_points, volume, dual_volume, gorenstein_idx, codimension)
    n = poly_n                  #...select number of vertices to ML, use '0' to mean all
    Pad_check = Pad             #...true to perform padding, false to select according to length
    Pad_choice = Pchoice        #...number to pad onto end of vectors, only relevant when padding
    GCD_scheme = gcd             #...whether to augment plucker coords by: (0) nothing, (1) pairwise gcds, (2) (n-1)-gcds
    Y_choice_range = Y_Ranges[Y_choice-3][1] - Y_Ranges[Y_choice-3][0] #... (max - min) for the selected varible to ML

    #Extract relevant X & Y data
    polygons_X, polygons_Y, last_poly_pts = [], [], []
    for idx, poly in enumerate(data):
        if int(poly[4]) == n or n == 0: #...extract only polygons with n vertices, or all polygons if n==0 (inefficient extra 'ifs' but this section is not the time bottleneck)
            if X_choice == 1 and poly[1]!=last_poly_pts: #...skip repeated lines in dataset (where plucker coords permuted)
                last_poly_pts = poly[1] #...keep track of last polygon, so know when moved onto next one
                polygons_X.append(list(chain(*literal_eval(poly[X_choice])))) #...if using vertices need to flatten to a vector                    
                polygons_Y.append(poly[Y_choice])
            elif X_choice == 2: 
                if GCD_scheme == 1:   #...augment vectors with pairwise gcds
                    polygons_X.append(literal_eval(poly[X_choice])+[np.gcd(*np.absolute(x)) for x in combinations(literal_eval(poly[X_choice]),2)])
                elif GCD_scheme == 2: #...augment vectors with (n-1)-gcds
                    polygons_X.append(literal_eval(poly[X_choice])+[np.gcd.reduce(np.absolute(x)) for x in combinations(literal_eval(poly[2]),poly[3]-1)])
                else: polygons_X.append(literal_eval(poly[X_choice]))
                polygons_Y.append(poly[Y_choice])
    number_polygon = len(polygons_Y)
    
    #Run inversion, editing data accordingly swapping param with last entry of the input vector
    if inversion_check:
        params = deepcopy(polygons_Y)
        for poly in range(len(polygons_X)):
            polygons_Y[poly] = polygons_X[poly][-1]
            polygons_X[poly][-1] = params[poly]
        Y_choice_range = max(polygons_Y)-min(polygons_Y)
        del(params)

    #Pad plucker coordinates if desired (only needed for n == 0)
    if Pad_check and n != 0:
        max_length = max(map(len,polygons_X))
        for polygon in polygons_X:
            while len(polygon) < max_length: #...pad all X vectors to the maximum length
                polygon += [Pad_choice]
        del(polygon,poly)

    #k-fold cross-val data setup
    k = k_cv #...number of cross-validations to perform
    tt_split = split #... only relevant if no cross-validation, sets the size of the train:test ratio split
    ML_data = [[polygons_X[index],polygons_Y[index]] for index in range(number_polygon)]

    np.random.shuffle(ML_data)
    Training_data, Training_values, Testing_data, Testing_values = [], [], [], []
    if k > 1:
        s = int(floor(len(ML_data)/k))        #...number of datapoints in each validation split
        for i in range(k):
            Training_data.append([HS[0] for HS in ML_data[:i*s]]+[HS[0] for HS in ML_data[(i+1)*s:]])
            Training_values.append([HS[1] for HS in ML_data[:i*s]]+[HS[1] for HS in ML_data[(i+1)*s:]])
            Testing_data.append([HS[0] for HS in ML_data[i*s:(i+1)*s]])
            Testing_values.append([HS[1] for HS in ML_data[i*s:(i+1)*s]])
    elif k == 1:
        s = int(floor(len(ML_data)*tt_split)) #...number of datapoints in train split
        Training_data.append([HS[0] for HS in ML_data[:s]])
        Training_values.append([HS[1] for HS in ML_data[:s]])
        Testing_data.append([HS[0] for HS in ML_data[s:]])
        Testing_values.append([HS[1] for HS in ML_data[s:]])

    #Create and Train NN
    #Define NN hyper-parameters
    def act_fn(x): return keras.activations.relu(x,alpha=0.01) #...leaky-ReLU activation
    number_of_epochs = 20           #...number of times to run training data through NN
    size_of_batches = 32            #...number of datapoints the NN sees per iteration of optimiser (high batch means more accurate param updating, but less frequently) 
    layer_sizes = [64,64,64,64]     #...number and size of the dense NN layers

    #Define lists to record training history and learning measures
    hist_data = []               #...training data (output of .fit(), used for plotting)
    metric_loss_data = []        #...list of learning measure losses [MAE,logcosh,MAPE,MSE]
    acc_list = []                #...list of accuracy ranges [\pm 0.5, \pm 5%, \pm 10%]

    #Train k independent NNs for k-fold cross-validation (learning measures then averaged over)
    for i in range(k):
        #Setup NN
        model = keras.Sequential()
        for layer_size in layer_sizes:
            model.add(keras.layers.Dense(layer_size, activation=act_fn))
            #model.add(keras.layers.Dropout(0.1)) #...dropout layer to reduce chance of overfitting to training data
        model.add(keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='logcosh') #...choose from: [MAE,MAPE,MSE,logcosh]
        #Train NN
        hist_data.append(model.fit(Training_data[i], Training_values[i], batch_size=size_of_batches, epochs=number_of_epochs, shuffle=True, validation_split=0., verbose=0))
        #Test NN
        predictions = np.ndarray.flatten(model.predict(Testing_data[i]))
        metric_loss_data.append([float(keras.losses.MAE(Testing_values[i],predictions)),float(keras.losses.logcosh(Testing_values[i],predictions)),float(keras.losses.MAPE(Testing_values[i],predictions)),float(keras.losses.MSE(Testing_values[i],predictions))])
        count_A, count_B, count_C = 0, 0, 0
        for test in range(len(predictions)):
            if Testing_values[i][test]-0.5 <= predictions[test] <= Testing_values[i][test]+0.5:
                count_A += 1
                count_B += 1
                count_C += 1
            elif Testing_values[i][test]-Y_choice_range*0.025 <= predictions[test] <= Testing_values[i][test]+Y_choice_range*0.025:
                count_B += 1
                count_C += 1
            elif Testing_values[i][test]-Y_choice_range*0.05 <= predictions[test] <= Testing_values[i][test]+Y_choice_range*0.05:
                count_C += 1
        acc_list.append([count_A/len(predictions),count_B/len(predictions),count_C/len(predictions)])

    #Output averaged testing metric accuracies and losses
    with open('./MLResults.txt','a') as myfile:
        myfile.write('Accuracies [\pm 0.5, \pm 0.025*range, \pm 0.05*range]: '+str(np.sum(acc_list,axis=0)/k)+'\nLosses [MAE, Log(cosh), MAPE, MSE]: '+str(np.sum(metric_loss_data,axis=0)/k))
    #print('\n####################################') #...uncomment to print measures as the investigations are running
    #print('Hyper-params:',headings[X_choice],'-->',headings[Y_choice],', with number vertices:',n,', for dataset size:',number_polygon,'polygons, with...',Pad_check,Pad_choice,Selection_len)
    #print('Average measures:')
    #print('[MAE, Log(cosh), MAPE, MSE]:',np.sum(metric_loss_data,axis=0)/k)
    #print('Accuracies [\pm 0.5, \pm 0.025*range, \pm 0.05*range]:',np.sum(acc_list,axis=0)/k)

    return [[X_choice,Y_choice,n],np.sum(acc_list,axis=0)/k,np.sum(metric_loss_data,axis=0)/k] #[[X_choice,Y_choice,n],Testing_values[-1],list(predictions)] #...optionally output true & predicted values for plotting

###############################################################################
#%% #Set-up file to write results to
with open('./MLResults.txt','w') as myfile:
    myfile.write('ML Results for Vert/Pluck --> Params, for n 3->6 in Dense LeakyReLU NN 4x64')

#Run ML for variety of investigation hyper-params
ML_results = [] #...save all investigation information
for x in input_set: 
    for y in output_set: 
        with open('./MLResults.txt','a') as myfile:
            myfile.write('\n\n#####################\nHyper-params: '+str(headings[x])+' --> '+str(headings[y]))
        for n in number_vertex_set:
            with open('./MLResults.txt','a') as myfile:
                myfile.write('\nn = '+str(n)+': ')
            if crossval_check:
                ML_results.append(ML(x,y,n,gcd=gcd_scheme))
            else:
                for tt in tt_ratio_set:
                    with open('./MLResults.txt','a') as myfile:
                        myfile.write('\nTrain proportion = '+str(tt)+': ')
                    ML_results.append(ML(x,y,n,gcd=gcd_scheme,k_cv=1,split=tt))

#Output the results in a raw format for data processing if desired
with open('./RawOutput.txt','w') as myfile2:
    myfile2.write(str(ML_results))

#print(ML_results) #...uncomment if wish to see all investigation information printed to current terminal
