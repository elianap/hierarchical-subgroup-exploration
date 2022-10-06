import operator

ops = {
    "<": operator.lt,
    "<=": operator.le,
    "=": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


def flip_values(
    df, indexes_to_flip, ratio=1, predicted_class_name="predicted", random_value=7
):
    import random

    random.Random(random_value).shuffle(indexes_to_flip)

    indexes_flip = indexes_to_flip[: int(len(indexes_to_flip) * ratio)]

    df.loc[indexes_flip, predicted_class_name] = (
        (~df[predicted_class_name].astype(bool)).astype(int).loc[indexes_flip]
    )
    return df, indexes_flip


def inject_set_to_value(
    df,
    value,
    indexes_to_flip,
    ratio=1,
    predicted_class_name="predicted",
    random_value=7,
):
    import random

    random.Random(random_value).shuffle(indexes_to_flip)

    indexes_flip = indexes_to_flip[: int(len(indexes_to_flip) * ratio)]

    df.loc[indexes_flip, predicted_class_name] = value
    return df, indexes_flip


class Injected_bias:
    def __init__(
        self,
        dict_attribute_values,
        ratio=1,
        noise=False,
        metric=None,
        id_exp=None,
        dataset_size=None,
    ):
        self.injected_bias_dict = dict_attribute_values
        self.list_injected_bias = [dict_attribute_values]
        self.ratio = ratio
        self.noise = noise
        self.size_injection = None
        self.indexes_bias = None
        self.flipped_indexes = None
        self.size_noise = 0
        self.metric = metric
        self.id_exp = id_exp if noise == False else f"{id_exp}_noise"
        self.dataset_size = None

    # Get the indexes of the pandas DataFrame matching the pattern
    def define_indexes_bias(self, df):
        indexes = df.index
        for attribute, rel_bias_attr_i in self.injected_bias_dict.items():
            for val_rels in rel_bias_attr_i:
                for rel, val in val_rels:

                    indexes = (
                        df.loc[indexes]
                        .loc[ops[rel](df.loc[indexes, attribute], val)]
                        .index
                    )

        # display(df.loc[indexes].loc[ops[rel](df.loc[indexes, attribute], val)])
        self.indexes_bias = list(indexes)
        return indexes

    def flip_values_injected_bias(
        self,
        df,
        indexes_to_flip=None,
        ratio=None,
        predicted_class_name="predicted",
        random_value=7,
    ):
        if indexes_to_flip is None:
            indexes_to_flip = self.indexes_bias
        if ratio is None:
            ratio = self.ratio
        df, self.flipped_indexes = flip_values(
            df,
            indexes_to_flip,
            ratio=ratio,
            predicted_class_name=predicted_class_name,
            random_value=random_value,
        )
        self.size_injection = len(self.flipped_indexes)
        return df

    def injected_bias_by_setting_a_value(
        self,
        df,
        value,
        indexes_to_flip=None,
        ratio=None,
        predicted_class_name="predicted",
        random_value=7,
    ):
        if indexes_to_flip is None:
            indexes_to_flip = self.indexes_bias
        if ratio is None:
            ratio = self.ratio
        df, self.flipped_indexes = inject_set_to_value(
            df,
            value,
            indexes_to_flip,
            ratio=ratio,
            predicted_class_name=predicted_class_name,
            random_value=random_value,
        )
        self.size_injection = len(self.flipped_indexes)
        return df

    def get_summary_dictionary_bias(self):
        return {
            "bias_pattern": self.injected_bias_dict,
            "ratio": self.ratio,
            "noise": self.noise,
            "size_injection": self.size_injection,
            "size_noise": self.size_noise,
            "id_exp": self.id_exp,
            "metric": self.metric,
            "size_dataset": self.dataset_size,
        }

    def __str__(self):
        return f"{self.injected_bias_dict}, noise: {self.noise}"
