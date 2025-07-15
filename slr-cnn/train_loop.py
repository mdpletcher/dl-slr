
import time

def main(
    d: data_setup.DataSetup,
    c: model_config.ModelConfig,
    model_name: str,
    epochs: int,
) -> None:
    start_loop = time.time()
    val_best_loss = 0.0
    val_labels, val_preds = [], []
    for epoch in range(epochs):
        start_epoch = time.time()
        t = timing.EpochTime(start_loop, start_epoch)
        c.model.train()
        train = train.Train(
            d,
            epoch,
            epochs,
            model_name,
            c
        )
        train.run()
        c.model.eval()
        val = validate.Validation(
            f,
            epoch,
            epochs,
            model_name,
            val_best_loss,
            c
        )
        val_best_loss = val.run()
        if epoch == epochs - 1:
            val_labels.append(val.epoch_labels)
            val_preds.append(val.epoch_preds)
        t.print_time_one_epoch()
    t.print_time_all_epochs()

        