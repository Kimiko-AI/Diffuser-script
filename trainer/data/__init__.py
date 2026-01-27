from .webdataset import get_wds_loader, get_fast_wds_loader

def get_dataloader(args, accelerator=None):
    """
    Factory function to get the dataloader based on configuration.
    Currently only supports 'webdataset' via get_wds_loader.
    """
    
    model_type = getattr(args, "model_type", "zimage")
    use_fast_mode = getattr(args, "fast_mode", False)
        
    if use_fast_mode:
        print("Using Fast WebDataset Loader (Random Resized Crop)")
        return get_fast_wds_loader(
            url_pattern=args.data_url,
            batch_size=args.train_batch_size,
            num_workers=getattr(args, "dataloader_num_workers", 8),
            is_train=True,
            resolution=getattr(args, "resolution", 256)
        )
    
    # Default to WebDataset with Bucketing
    return get_wds_loader(
        url_pattern=args.data_url,
        batch_size=args.train_batch_size,
        num_workers=getattr(args, "dataloader_num_workers", 8),
        is_train=True,
        base_resolution=getattr(args, "resolution", 256),
        bucket_step_size=getattr(args, "bucket_step_size", 32)
    )