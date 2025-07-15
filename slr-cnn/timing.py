import time

class EpochTime:
    """Time one or all epochs"""

    def __init__(self, start_loop, start_epoch):
        self.start_loop = start_loop
        self.start_epoch = start_epoch

    def print_time_all_epochs(self) -> None:
        """Print time for all epochs to complete"""
        elapsed_time = time.time() - self.start_loop
        print(
            "Finished all epochs in %02d min, %02d sec" % (
                elapsed_time // 60, elapsed_time % 60
            )
        )
    
    def print_time_one_epoch(self) -> None:
        """Print time for a single epoch to complete"""
        elapsed_time = time.time() - self.start_epoch
        print(
            "Epoch complete in %02d min, %02d sec" % (
                elapsed_time // 60, elapsed_time % 60
            )
        )