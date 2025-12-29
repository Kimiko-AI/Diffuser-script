from .webdataset import get_wds_loader, get_fast_wds_loader

def get_dataloader(args, accelerator=None):
    """
    Factory function to get the dataloader based on configuration.
    Currently only supports 'webdataset' via get_wds_loader.
    """
    
    model_type = getattr(args, "model_type", "zimage")
    use_fast_mode = getattr(args, "fast_mode", False)
    
    # Default to fast mode for sr_dit unless explicitly disabled (if someone were to add a disable flag)
    # or just use the fast_mode flag which defaults to False for others.
    if model_type == "sr_dit" and not getattr(args, "disable_fast_mode", False):
        use_fast_mode = True
        
    if use_fast_mode:
        print("Using Fast WebDataset Loader (Center Crop 256x256)")
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
