import numpy as np
from collections import Counter


class Tree:
    
    class Node:
        def __init__(self, feature_idx=None, value=None, left=None, right=None,*,label=None):
            self.feature_idx = feature_idx
            self.value = value
            self.left_node = left
            self.right_node = right
            self.label = label

            
    def __init__(self, min_split=2, max_depth=100, n_feature = None):
        self.min_split=2
        self.max_depth = max_depth
        self.n_feature = n_feature
        self.root  = None
        
    def fit(self, X, y):
        _ , total_feature = X.shape
        self.n_feature = total_feature if self.n_feature == None else self.n_feature
        self.root = self.make_node(X,y)
        
         
    def make_node(self, X, y, depth=0):
        n_sample, total_feature = X.shape
        n_labels = len(np.unique(y))
        
        #checking leaf node
        if (depth >= self.max_depth) or n_labels==1 or n_sample < self.min_split:

            common_label = Counter(y).most_common(1)[0][0]

            return Tree.Node(label=common_label)
            
        #if not leaf, then make parent and child nodes
        feature_idxs = np.random.choice(total_feature, self.n_feature, False) 
        #take randomly some columns out of the total # column of X 
        #getting the random subdataset
        
        
        #initialize optimized variables
        best_gaining = 0
        split_feature_idx = None
        split_value = None
        
        #start looping through random selected features, to find the best feature with best threshold
        for feature_idx in feature_idxs:
            feature = X[:,feature_idx]
            unique_values= np.unique(feature)
            #for each feature, loop through unique thresholds, to find the best threshold
            for unique_value in unique_values:
                gaining = self.info_gain(y, feature, unique_value)
                if gaining >= best_gaining:
                    best_gaining = gaining
                    split_feature_idx = feature_idx
                    split_value = unique_value
                
        #spliting sample
        left_idxs =  np.argwhere(X[:,split_feature_idx] <= split_value).flatten()
        right_idxs = np.argwhere(X[:,split_feature_idx] > split_value).flatten()
        
        left_sample_X = X[left_idxs,:]
        left_sample_y = y[left_idxs]
        right_sample_X = X[right_idxs,:]
        right_sample_y = y[right_idxs]
        
        left_node = self.make_node(left_sample_X, left_sample_y, depth+1)
        right_node = self.make_node(right_sample_X, right_sample_y, depth+1)
        
        return Tree.Node(split_feature_idx, split_value, left_node, right_node)
      
    
    def entropy(self, y):
        py = np.bincount(y)/len(y)
        py = np.array([p for p in py if p>0]) # for extreme cases of label has no count, log2(0) -> inf
        E = np.sum(-py*np.log2(py)) 
        #E = np.sum([-p * np.log2(p) for p in py])
        return E     
    
    def info_gain(self, y, feature, split_value):
        parent_E = self.entropy(y)
        child1_idxs = np.argwhere(feature <= split_value).flatten()
        child2_idxs = np.argwhere(feature > split_value).flatten()
        
        
        child1_E = self.entropy(y[child1_idxs])
        child2_E = self.entropy(y[child2_idxs])
        gaining = parent_E - (len(child1_idxs)*child1_E + len(child2_idxs)*child2_E)/len(y)
        return gaining
        
    def traverse(self, x, node):
        if node.label != None:
            return node.label
        if x[node.feature_idx] > node.value:
            return self.traverse(x, node.right_node)
        else:
            return self.traverse(x, node.left_node)
        
    def predict(self, X):
        return np.array([self.traverse(x, self.root) for x in X])
        
    def accuracy(self, X, y):
        y_hat = self.predict(X)
        score = np.sum(y_hat==y)/len(y)
        return f'{score:.3f}'