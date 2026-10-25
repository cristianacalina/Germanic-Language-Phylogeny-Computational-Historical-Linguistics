from topologies import topologies
import pandas as pd
import math


class Node():
    def __init__(self, name, age, children=None):
        self.name = name
        self.age = age
        self.children = children
        self.is_leaf = True


def build_tree(tree, ages):
    language = tree["name"]
    if language in ages: # leaves don't have ages (except Gothic)
        age = ages[language]
    else:
        age = 0
    parent_node = Node(language, age)
    if "children" in tree.keys():
        parent_node.is_leaf = False
        parent_node.children = []
        for child in tree["children"]:
            child_node = build_tree(child, ages) # RECURSION LOOP HERE
            parent_node.children.append(child_node)
    return parent_node # we return the root node which has all its children and their children linked


def get_likelihoods(node, binary_cognates, similarities):

    # if it's a leaf we can directly observe the data and get the actual likelihoods for the current node
    if node.is_leaf:
        observed_state = binary_cognates[1][node.name]
        if observed_state == -1:
            # Missing data: standard Felsenstein treatment is uninformative (1,1).
            # The tip contributes a factor of 1 and drops out of the product.
            likelihood_state_0 = 1.
            likelihood_state_1 = 1.
        elif observed_state == 0:
            likelihood_state_0 = 1.
            likelihood_state_1 = 0.
        else:
            likelihood_state_0 = 0.
            likelihood_state_1 = 1.
        return likelihood_state_0, likelihood_state_1

    # get language similarity in case we have missing data
    #print(node.__dict__)
    childA = node.children[0].name
    childB = node.children[1].name
    if childA.startswith("Proto") or childB.startswith("Proto"):
        similarity = 1
    else:
        condition1a = similarities["Language 1"] == childA
        condition1b = similarities["Language 2"] == childA
        condition2a = similarities["Language 1"] == childB
        condition2b = similarities["Language 2"] == childB
        df = similarities[(condition1a | condition1b) & (condition2a | condition2b)].reset_index(drop=True)
        similarity = df.iat[0, 2]

    # RECURSION LOOPS HERE
    childA_likelihood_if_0, childA_likelihood_if_1 = get_likelihoods(node.children[0], binary_cognates, similarities)
    childB_likelihood_if_0, childB_likelihood_if_1 = get_likelihoods(node.children[1], binary_cognates, similarities)

    childA_likelihood_if_0, childA_likelihood_if_1, childB_likelihood_if_0, childB_likelihood_if_1 = check_update_probabilities(childA_likelihood_if_0, childA_likelihood_if_1, childB_likelihood_if_0, childB_likelihood_if_1, similarity)

    branch_length_A = node.age - node.children[0].age
    
    A_given_0, A_given_1 = get_child_probabilities(branch_length_A, childA_likelihood_if_0, childA_likelihood_if_1)
    branch_length_B = node.age - node.children[1].age
    B_given_0, B_given_1 = get_child_probabilities(branch_length_B, childB_likelihood_if_0, childB_likelihood_if_1)

    likelihood_state_0 = A_given_0 * B_given_0
    likelihood_state_1 = A_given_1 * B_given_1
    return likelihood_state_0, likelihood_state_1


def check_update_probabilities(childA_0, childA_1, childB_0, childB_1, similarity):
    if childA_0 == -1 and childB_0 == -1:
        childA_0 = 0.5
        childA_1 = 0.5
        childB_0 = 0.5
        childB_1 = 0.5
    else:
        if childA_0 == -1:
            childA_0 = similarity * childB_0
            childA_1 = similarity * childB_1
        if childB_0 == -1:
            childB_0 = similarity * childA_0
            childB_1 = similarity * childA_1
    return childA_0, childA_1, childB_0, childB_1


def get_child_probabilities(branch_length, prob0, prob1):
    if branch_length <= 0: # in case tree is impossible (ex: Gothic older than proto age)
        return 0, 0
    r = 0.805                     # glottochronology retention constant
    alpha = 0.5 / 1000                  # change alpha value
    beta = -math.log(r) / 1000   
    s = alpha + beta
    e = math.exp(-s * branch_length)
    _0_to_0 = beta / s + alpha / s * e
    _0_to_1 = alpha / s - alpha / s * e
    _1_to_0 = beta / s - beta / s * e
    _1_to_1 = alpha / s + beta / s * e
    child_given_0 = _0_to_0 * prob0 + _0_to_1 * prob1
    child_given_1 = _1_to_0 * prob0 + _1_to_1 * prob1
    return child_given_0, child_given_1


def load_cognates(cognate_file):
    cognates = pd.read_csv(cognate_file, sep="\t", index_col=0) # get the cognates data into a pandas dataframe
    cognates = cognates.fillna(value=-1).astype(int) # replace all missing values by -1
    column_names = [col.split(" ")[0].strip(":") for col in cognates.columns] # get the clean column names
    cognates.columns = column_names # replace the column names with the clean ones
    cognates.to_csv("data_cognates.tsv", sep="\t")
    return cognates


def make_binary_cognates(cognates):
    # Dictionary of classes and their meanings
    class_to_meaning = {}
    for meaning, row in cognates.iterrows():
        for value in row.values:
            if value != -1:
                class_to_meaning[value] = meaning
    # Set of all unique class IDs
    cognates_array = cognates.values.ravel()
    unique_classes = set(cognates_array) - {-1}  # gets rid of missing data
    # Binary row for each class
    rows = []
    row_labels = []
    for cognate_class in unique_classes:
        meaning = class_to_meaning[cognate_class]
        # Make binary row
        binary_row = {}
        for language in cognates.columns:
            cell = cognates.loc[meaning, language]
            if cell == -1:
                binary_row[language] = -1
            elif cell == cognate_class:
                binary_row[language] = 1
            else:
                binary_row[language] = 0
        rows.append(binary_row)
        row_labels.append(f"{meaning}_{cognate_class}")
    # Create a Panda of binary cognates
    binary_cognates = pd.DataFrame(rows, index=row_labels)
    binary_cognates.to_csv("data_binary_cognates.tsv", sep="\t")
    return binary_cognates


def calculate_similarities(df):
    languages = df.columns
    results = []
    for i in range(len(languages)):
        for j in range(i + 1, len(languages)):
            Lang1 = languages[i]
            Lang2 = languages[j]
            col1 = df[Lang1]
            col2 = df[Lang2]
            # only look at meanings where BOTH languages have data
            both_present = (col1 != -1) & (col2 != -1)
            matches = ((col1 == col2) & both_present).sum()
            total = both_present.sum()
            similarity = matches / total if total > 0 else 0
            results.append([Lang1, Lang2, similarity])
    results_df = pd.DataFrame(results, columns = ["Language 1", "Language 2", "Similarity"])
    results_df = results_df.sort_values(by="Similarity", ascending = False)
    results_df.to_csv("data_similarities.tsv", sep="\t")
    return results_df

def make_list_of_all_trees(topology_file):
    trees = []
    for topology_data in topology_file.values():
        structure = topology_data["tree"]
        for age_combination in topology_data["age combinations"].values():
            age_combination["Gothic"] = 1650 # Gothic is a dead language so even as a leaf it has an age, which is given in the instructions
            tree_from_root = build_tree(structure, age_combination)
            # each tree is technically just the root node,
            # but all the children nodes are linked to their parent's children list,
            # so you can access all nodes from the root by going deeper and deeper
            trees.append(tree_from_root)
    return trees


def calculate_tree_log_likelihoods(tree_list, binary_cognates, similarities):
    r = 0.805
    alpha = 0.5 / 1000    # change alpha value        
    beta = -math.log(r) / 1000
    if alpha == 0.0:
        initial_prob_state_0, initial_prob_state_1 = 0.5, 0.5   # Dollo fallback
    else:
        initial_prob_state_0 = beta / (alpha + beta)
        initial_prob_state_1 = alpha / (alpha + beta)
    tree_log_likelihoods = []
    for i in range(len(tree_list)):
        tree = tree_list[i]
        cognate_log_likelihoods = []
        for binary_cognate_row in binary_cognates.iterrows():
            likelihood_state_0, likelihood_state_1 = get_likelihoods(tree, binary_cognate_row, similarities)
            cognate_likelihood = initial_prob_state_0 * likelihood_state_0 + initial_prob_state_1 * likelihood_state_1
            if cognate_likelihood > 0:
                cognate_log_likelihoods.append(math.log(cognate_likelihood))
            else:
                cognate_log_likelihoods.append(float("-inf"))
        final_log_likelihood = sum(cognate_log_likelihoods)
        tree_log_likelihoods.append(final_log_likelihood)
    return tree_log_likelihoods

def make_data_easier_to_analise(log_likelihoods):
    max_log_likelihood = max(log_likelihoods)
    relative_likelihoods = []
    delta_log_likelihoods = []
    for log_likelihood in log_likelihoods:
        relative_likelihoods.append(math.exp(log_likelihood - max_log_likelihood))
        delta_log_likelihoods.append(log_likelihood - max_log_likelihood)
    return relative_likelihoods, delta_log_likelihoods


def main():
    cognate_file = "chl2026germanic_cognacy.tab"
    topology_file = topologies

    cognates = load_cognates(cognate_file) # LOAD THE COGNATES DATA INTO A PANDAS DATAFRAME
    binary_cognates = make_binary_cognates(cognates) # MAKE BINARY COGNATE CODING, ALSO A PANDAS DATAFRAME
    similarities = calculate_similarities(cognates) # CALCULATE LANGUAGE SIMILARITY PER LANGUAGE PAIR, ALSO A PANDAS DATAFRAME
    trees = make_list_of_all_trees(topology_file) # BUILD THE TREES AS INSTANCES OF THE NODES CLASS
    tree_log_likelihoods = calculate_tree_log_likelihoods(trees, binary_cognates, similarities) # CALCULATE LIKELIHOODS 
    relative_likelihoods, delta_log_likelihoods = make_data_easier_to_analise(tree_log_likelihoods) # TRANSFORM LOG_LIKELIHOODS INTO DELTA_LOG AND RELATIVE LIKELIHOODS

    alpha_label = "0.5"   # change to match the alpha you're running
    with open(f"results_{alpha_label}.tsv", "w") as results:
        results.write("Tree\t"
                      "LogLikelihood\t"
                      "DeltaLogLikelihood\t"
                      "RelativeLikelihood\n")
        for i in range(len(trees)):
            result = (
                f"Tree {i}\t"
                f"{tree_log_likelihoods[i]}\t"
                f"{delta_log_likelihoods[i]}\t"
                f"{relative_likelihoods[i]}\n"
            )
            results.write(result)

if __name__ == "__main__":
    main()
