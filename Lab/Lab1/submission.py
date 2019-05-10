## import modules here 

################# Question 0 #################

def add(a, b):  # do not change the heading of the function
    return a + b


################# Question 1 #################

def nsqrt(x):  # do not change the heading of the function
    if x <= 1:
        return x
    root = x
    while root > x / root:
        root = (root + x / root) // 2
    return int(root)


################# Question 2 #################


# x_0: initial guess
# EPSILON: stop when abs(x - x_new) < EPSILON
# MAX_ITER: maximum number of iterations

## NOTE: you must use the default values of the above parameters, do not change them

def find_root(f, fprime, x_0=1.0, EPSILON=1E-7, MAX_ITER=1000):  # do not change the heading of the function
    xplus1 = x_0
    for i in range(0, MAX_ITER):
        MAX_ITER -= 1
        x = xplus1
        xplus1 = xplus1 - f(xplus1) / fprime(xplus1)
        if abs(xplus1 - x) <= EPSILON:
            return xplus1
    return xplus1


################# Question 3 #################

class Tree(object):
    def __init__(self, name='ROOT', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)


def make_tree(tokens):  # do not change the heading of the function
    new_tree = Tree(tokens[0])
    parent_node = new_tree
    child_node = new_tree
    raw_tree = []

    for i in range(1, len(tokens)):
        if tokens[i] == '[':
            raw_tree.append(parent_node)
            parent_node = child_node
            i += 1
            continue
        elif tokens[i] == ']':
            i += 1
            parent_node = raw_tree.pop()
            continue
        child_node = Tree(tokens[i])
        parent_node.add_child(child_node)
        i += 1
    return new_tree


def max_depth(root):  # do not change the heading of the function
    if root.children == None:
        return 1
    else:
        result = [1]
        for i in root.children:
            result.append(max_depth(i) + 1)
    return max(result)
