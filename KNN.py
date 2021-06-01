import csv
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
trainingdata=[]
testdata=[]
def loaddata(trainingdata=[],testdata=[]):
    with open("C:\\Users\\vipul chawla\\Desktop\\irisproject.txt") as csvfile:
        lines=csv.reader(csvfile,delimiter='\t')
        dataset=list(lines)
        for i in range(len(dataset)):
            for j in range(4):
                dataset[i][j]=float(dataset[i][j])
            if(i in range(30) or i in range(50,80) or i in range(110,150)):
                trainingdata.append(dataset[i])
            else:
                testdata.append(dataset[i])
loaddata(trainingdata,testdata)
def euclidist(instance1,instance2,length):
    distance=0
    for i in range(length):
        distance+=pow((instance1[i]-instance2[i]),2)
    return math.sqrt(distance)
def getkclosestneighbours(trainingdata,testinstance,k):
    distances=[]
    neighbours=[]
    for x in range(len(trainingdata)):
        dist=euclidist(trainingdata[x],testinstance,4)
        distances.append((trainingdata[x],dist))
    def sortSecond(val):
        return val[1]
    distances.sort(key=sortSecond)
    for i in range(k):
        neighbours.append(distances[i][0])
    return neighbours
def getlabel(neighbours):
    votes={}
    for i in range(len(neighbours)):
        response=neighbours[i][-1]
        if response in votes:
            votes[response]+=1
        else:
            votes[response]=1
    def maxi(votes):
        return votes[1]
    s=sorted(votes,key=maxi,reverse=True)
    return s[0]
predicted=testdata
def error(testdata,k):
    count=0
    print("    Predicted   Actual")
    for i in range(len(testdata)):
        print(len(testdata))
        neighbours=getkclosestneighbours(trainingdata,testdata[i],k)
        temp=getlabel(neighbours)
        print(f"{i+1}   {temp}      {testdata[i][-1]}")
        if(temp==testdata[i][-1]):
            count+=1
        else:
            predicted[i][-1]=temp
    percentage=(count/float(len(testdata)))*100
    return percentage
print("Trainingcases: ",len(trainingdata),"\nTestcases: ",len(testdata))
k=int(input("Enter the value of k\n"))
print(f"Percentage of correct output while taking k={k} is: {error(testdata,k)}")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x1=[]
y1=[]
z1=[]
c1=[]
x2=[]
y2=[]
z2=[]
c2=[]
x3=[]
y3=[]
z3=[]
c3=[]
x=[]
y=[]
z=[]
c=[]
for i in range(30):
    x1.append(trainingdata[i][0])
    y1.append(trainingdata[i][1])
    z1.append(trainingdata[i][2])
    c1.append(trainingdata[i][3])
img = ax.scatter(x1, y1, z1, c=c1, cmap=plt.hot(),marker="d",edgecolor='g',label="setosa train_exa")
for i in range(30,60):
    x2.append(trainingdata[i][0])
    y2.append(trainingdata[i][1])
    z2.append(trainingdata[i][2])
    c2.append(trainingdata[i][3])
img = ax.scatter(x2, y2, z2, c=c2, cmap=plt.hot(),marker="*",edgecolor='b',label="versicolor train_exa")
for i in range(60,100):
    x3.append(trainingdata[i][0])
    y3.append(trainingdata[i][1])
    z3.append(trainingdata[i][2])
    c3.append(trainingdata[i][3])
img = ax.scatter(x3, y3, z3, c=c3, cmap=plt.hot(),marker="o",edgecolor='r',label="verginica train_exa")
for i in range(len(predicted)):
    x.append(predicted[i][0])
    y.append(predicted[i][1])
    z.append(predicted[i][2])
    c.append(predicted[i][3])
img = ax.scatter(x, y, z, c=c, cmap=plt.hot(),marker="^",edgecolor='y',label="Predicted Results",s=50)
fig.colorbar(img)
ax.set_xlim3d(0,5.6)
ax.set_ylim3d(0,5.6)
ax.set_zlim3d(0,5.6)
ax.set_xlabel("Sepal_length")
ax.set_ylabel("Sepal_width")
ax.set_zlabel("petal_length")
ax.set_title(f"Training data and Predicted results for k={k}")
plt.legend()
plt.show()