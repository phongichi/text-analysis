import numpy as np               #for maths
import pandas as pd              #for data manipulation
import matplotlib.pyplot as plt  #for visualization



#Load the data
#data 
path = r'../input/NationalNames.csv'
data = pd.read_csv(path)

#get names from the dataset
data['Name'] = data['Name']

#get first 10000 names
data = np.array(data['Name'][:10000]).reshape(-1,1)

#covert the names to lowee case
data = [x.lower() for x in data[:,0]]

data = np.array(data).reshape(-1,1)

#print("Data Shape = {}".format(data.shape))
#print()
#print("Lets see some names : ")
#print(data[1:10])


#Transform data to equal length by adding -- > ('.') dots :

#to store the transform data
transform_data = np.copy(data)

#find the max length name
max_length = 0
for index in range(len(data)):
    max_length = max(max_length,len(data[index,0]))

#make every name of max length by adding '.'
for index in range(len(data)):
    length = (max_length - len(data[index,0]))
    string = '.'*length
    transform_data[index,0] = ''.join([transform_data[index,0],string])


#print("Transformed Data")
#print(transform_data[1:10])


#Lets Make Vocabulary
#to store the vocabulary
vocab = list()

for name in transform_data[:,0]:
    vocab.extend(list(name))

vocab = set(vocab)
vocab_size = len(vocab)

#print("Vocab size = {}".format(len(vocab)))
#print("Vocab      = {}".format(vocab))


# Map characters to ids and ids to characters ( text - number)

#map char to id and id to chars
char_id = dict()
id_char = dict()

for i,char in enumerate(vocab):
    char_id[char] = i
    id_char[i] = char

#print('a-{}, 22-{}'.format(char_id['a'],id_char[22]))


# Make the Train dataset
# list of batches of size = 20
train_dataset = []

batch_size = 20

#split the trasnform data into batches of 20
for i in range(len(transform_data)-batch_size+1):
    start = i*batch_size
    end = start+batch_size
    
    #batch data
    batch_data = transform_data[start:end]
    
    if(len(batch_data)!=batch_size):
        break
        
    #convert each char of each name of batch data into one hot encoding
    char_list = []
    for k in range(len(batch_data[0][0])):
        batch_dataset = np.zeros([batch_size,len(vocab)])
        for j in range(batch_size):
            name = batch_data[j][0]
            char_index = char_id[name[k]]
            batch_dataset[j,char_index] = 1.0
     
        #store the ith char's one hot representation of each name in batch_data
        char_list.append(batch_dataset)
    
    #store each char's of every name in batch dataset into train_dataset
    train_dataset.append(char_list)
















