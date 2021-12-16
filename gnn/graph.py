def gen_feature_vec_template(pset):
    prim_list = pset.primitives[object]
    args = pset.arguments

    possible = [p.name for p in prim_list]
    possible.extend(args)
    return possible


def node_features(node, template):
    vec = []
    found = False
    # function
    for name in template:
        if node.name == name:
            vec.append(1)
            found = True
        else:
            vec.append(0)

    # constant terminal
    if found:
        vec.append(0)
    else:
        vec.append(node.value)

    return vec


def graph_from_tree(individual, feature_template):
    stack = []
    for node in reversed(individual):
        features = node_features(node, feature_template)
        node_entry = {
            'features': features,
            'children': []
        }

        if len(stack) != node.arity:
            stack.append(node_entry)
        else:
            node_entry['children'] = stack[-node.arity:]
            stack = stack[:-node.arity]
            stack.append(node_entry)

    assert len(stack) == 1, "Invalid tree"
    return stack[0]
