from .webdataset import get_wds_loader

def get_dataloader(args, accelerator=None):
    """
    Factory function to get the dataloader based on configuration.
    Currently only supports 'webdataset' via get_wds_loader.
    """
    # In the future, check args.dataset_type or similar
    # dataset_type = getattr(args, "dataset_type", "webdataset")
    
    # Default to WebDataset for now as it's the only implementation
    return get_wds_loader(
        url_pattern=args.data_url,
        batch_size=args.train_batch_size,
        num_workers=getattr(args, "dataloader_num_workers", 8),
        is_train=True,
        base_resolution=getattr(args, "resolution", 256),
        bucket_step_size=getattr(args, "bucket_step_size", 32)
    )
