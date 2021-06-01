import pandas as pd 
import math
df = pd.read_csv(r"C:\\Users\\vipul chawla\\Downloads\\bank.csv")
titles = list(df.columns)
titles[7],titles[-1] = titles[-1],titles[7]
df = df[titles]

training_data = []
test_data = []
df_list = df.values.tolist()

def loaddata():
	for i in range(len(df_list)):
		if(i < (len(df_list)*(0.78))):     ## 0.78 is for training data portion from total data.
			training_data.append(df_list[i])
		else:
			test_data.append(df_list[i])

loaddata()

def unique_val(rows,col):                 ## To find different classes in a column
	return set([row[col] for row in rows])
# print(unique_val(df_list,0))


def class_counts(rows):
	counts = {}								## Creates a empty dictionary for class type -> count
	for row in rows:
		if(row[-1] in counts):
			counts[row[-1]] += 1
		else:
			counts[row[-1]] = 1
	return counts
# print(class_counts(df_list,3))


def is_numeric(value):
	return isinstance(value,int) or isinstance(value, float)   ## Returns true is value is either float or int


class Question():

	def __init__(self,column,value):
		self.column = column
		self.value = value

	def match(self,example):
		val = example[self.column]
		if is_numeric(val):
			return val >= self.value
		else:
			return val == self.value

	def __repr__(self):
		condition = " == "
		if is_numeric(self.value):
			condition = ">="
		s = "Is " + titles[self.column] + " " + condition + " "+ str(self.value) + "?"
		return s

# q = Question(2,"Marital")
# print(q.match(df_list[1]))
# print(q.repr())


def entropy(rows):
	dict = class_counts(rows)
	counts = list(dict.values())
	e = 0
	T = sum(counts)
	if(len(counts)==1):
		return 0
	for i in range(len(counts)):
		t = counts[i]
		e -= float((t/T)*math.log(t/T,2))
	return e
# Te = entropy(training_data,7)

def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def info_gain(left,right,t_entropy):
	p = len(left) / (len(left) + len(right))
	return t_entropy - p*entropy(left) - (1-p)*entropy(right)


def find_best_split(rows):
   
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    t_entropy = entropy(rows)
    n_features = len(titles) - 1  # number of columns (without label column)

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  
            question = Question(col, val)

            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows, false_rows, t_entropy)
													# Calculates the information gain from this split     
            
            if gain > best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf():
	def __init__(self,rows):
		self.predictions = class_counts(rows)


class Decision_Node():
	def __init__(self,question,true_branch,false_branch):
		self.question = question
		self.true_branch = true_branch
		self.false_branch = false_branch


def build_tree(rows):

	gain,question = find_best_split(rows)

	if (gain == 0):
		return Leaf(rows)

	true_rows,false_rows = partition(rows,question)
	true_branch = build_tree(true_rows)
	false_branch = build_tree(false_rows)

	return Decision_Node(question,true_branch,false_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

my_tree = build_tree(training_data)
# print_tree(my_tree)
def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

temp = 0
def accuracy():
	global temp
	for row in test_data:
		dict = print_leaf(classify(row,my_tree))
		x = list(dict.keys())
		if(row[-1] == x[0]):
			temp += 1
	return (temp/len(test_data))*100
print("Accuracy " ,accuracy())
