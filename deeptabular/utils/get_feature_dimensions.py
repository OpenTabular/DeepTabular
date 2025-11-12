def get_feature_dimensions(num_feature_info, cat_feature_info, embedding_info):
    input_dim = 0
    for _, feature_info in num_feature_info.items():
        input_dim += feature_info["dimension"]
    for _, feature_info in cat_feature_info.items():
        input_dim += feature_info["dimension"]
    for _, feature_info in embedding_info.items():
        input_dim += feature_info["dimension"]

    return input_dim
