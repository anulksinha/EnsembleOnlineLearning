import numpy as np
########################################################################################
##################### Reading Files ####################################################
with open ("sampletext.txt", "r") as myfile:
    data_1=myfile.read()
with open ("sampletext1.txt", "r") as myfile:
    data_2=myfile.read()
with open ("sampletext2.txt", "r") as myfile:
    data_3=myfile.read()
with open ("sampletext3.txt", "r") as myfile:
    data_4=myfile.read()
fulldata = data_1+data_2+data_3
print(len(fulldata))
##################### Getting Characters ###############################################
charlist = sorted(list(set(fulldata)))
#print(charlist)
charlist_to_intlist = dict((ch, il) for il, ch in enumerate(charlist))
intlist_to_charlist = dict((il, ch) for il, ch in enumerate(charlist))
print(charlist_to_intlist['.'])
#print('length '+str(len(data)))
test='this is a test. Please ignore'
encoded_data_1=[charlist_to_intlist[d] for d in data_1]
encoded_data_2=[charlist_to_intlist[d] for d in data_2]
encoded_data_3=[charlist_to_intlist[d] for d in data_3]
encoded_data_4=[charlist_to_intlist[d] for d in data_4]
encoded_data_full=[charlist_to_intlist[d] for d in fulldata]
###################### number of characters per line ###################################
num=20
X1=np.zeros((len(encoded_data_1)-num,num))
X2=np.zeros((len(encoded_data_2)-num,num))
X3=np.zeros((len(encoded_data_3)-num,num))
X4=np.zeros((len(encoded_data_4)-num,num))
X_full=np.zeros((len(encoded_data_full)-num,num))


################### Creating 20 character sequences ####################################
for i in range(len(encoded_data_1)-num):
    X1[i,0:num]=encoded_data_1[i:i+num]
for i in range(len(encoded_data_2)-num):
    X2[i,0:num]=encoded_data_2[i:i+num]
for i in range(len(encoded_data_3)-num):
    X3[i,0:num]=encoded_data_3[i:i+num]
for i in range(len(encoded_data_4)-num):
    X4[i,0:num]=encoded_data_4[i:i+num]
for i in range(len(encoded_data_full)-num):
    X_full[i,0:num]=encoded_data_full[i:i+num]
#print(X.shape)
################### Saving array in files ##############################################
np.savetxt('s1.txt', X1,fmt='%1i')
np.savetxt('s2.txt', X2,fmt='%1i')
np.savetxt('s3.txt', X3,fmt='%1i')
np.savetxt('s4.txt', X4,fmt='%1i')
np.savetxt('s_full.txt', X_full,fmt='%1i')