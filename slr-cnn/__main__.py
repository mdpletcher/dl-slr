


def model_setup(
    f: cocpit.fold_setup.FoldSetup, 
    model_name: str, 
    epochs: int
) -> None:
    """
    Create instances for model configurations and training/validation. Runs model.

    Args:
        f (cocpit.fold_setup.FoldSetup): instance of FoldSetup class
        model_name (str): name of model architecture
        epochs (int): number of iterations on dataset
    """
    m = models.SLR_CNN()
    # call method based on str model name

    c = model_config.ModelConfig(m.model)
    c.set_optimizer()
    c.set_criterion()
    c.to_device()
    runner.main(
        f,
        c,
        model_name,
        epochs,
        kfold=0,
    )
