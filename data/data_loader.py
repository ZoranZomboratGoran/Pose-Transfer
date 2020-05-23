
def CreateDataLoader(opt, phase, pairList, dset):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt, phase, pairList, dset)
    return data_loader
