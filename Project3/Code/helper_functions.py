def get_std(pandas_object):
    metadata_std = [x.std() for x in pandas_object.values.T]

    return metadata_std


def get_corr(pandas_object, list_of_features=[]):
    if len(list_of_features) == 0:
        list_of_features = pandas_object.columns

    corr_list = []

    for i in list_of_features:
        corr_list.append([])
        for j in list_of_features:
            corr_list[-1].append(pandas_object[i].corr(pandas_object[j]))

    return corr_list
